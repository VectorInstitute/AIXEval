"""Judge runners: tie a judge_* function to a sealed record, never mutating it.

``run_grounding_judge`` and ``run_trajectory_optimality_judge`` read
the original dump, not just the sealed record: ingest's copy-only
rule already stripped evidence/trajectory text out of it.
``run_abstention_judge`` and ``run_refusal_judge`` need no dump at
all: their only input, ``output``, already lives on the sealed
record. All four return a new record plus a separate rationale
attachment, never mutate the input record, and always return an
attachment on success or failure, so a caller can see why a record
has no verdict rather than just a silently missing field.

``tool_call_accuracy`` gets two independent runners that both write
the same ``tool_call_accuracy_judge`` field:
``run_tool_call_accuracy_check`` (deterministic, no LLM, compares
the dump's actual tool calls against a declared ``gold_tool_calls``
list) and ``run_tool_call_accuracy_judge`` (LLM-based, same shape as
``run_trajectory_optimality_judge`` plus a ``gold_tool_plan`` read off
the record). See ``aixeval.scorecard.tool_call_accuracy`` for why
they're interchangeable.

``toxicity`` also gets two independent mechanisms writing the same
``toxicity_judge`` field, but only one of them is a runner here:
``run_toxicity_judge`` (LLM-based, same no-dump-needed shape as
``run_abstention_judge``/``run_refusal_judge``). The other mechanism,
a classifier call, needs no raw dump either and is cheap enough to
call inline; it lives directly in
``aixeval.scorecard.toxicity.score_toxicity``, not as a runner
here. See that module for why the two are interchangeable.

``run_pairwise_preference_judge`` is the first runner whose judge
makes two model calls per verdict, not one: position-bias
correction (see ``aixeval.judge.pairwise_preference``), averaging
the judged preference over both presentation orders of ``output`` and
a declared ``baseline_output``. It uses ``_ListCapturingProvider``, not
``CapturingProvider``, so the attachment can carry both raw replies,
not just the last one.

``collect_rerun_scores`` is not a runner itself; it calls any one of
the runners above ``n`` times against the same, unchanged input and
collects the resulting scores, for ``aixeval.scorecard.
test_retest_stability``'s declared ``rerun_scores`` field. Optional:
nothing else in this module depends on it.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from aixeval.ingest.langfuse import ingest_langfuse
from aixeval.ingest.load import load_json_object
from aixeval.ingest.record import validate_record
from aixeval.judge.abstention import PROMPT_VERSION as ABSTENTION_PROMPT_VERSION
from aixeval.judge.abstention import judge_abstention
from aixeval.judge.config import build_client_from_env
from aixeval.judge.cross_modal_hallucination import PROMPT_VERSION as CROSS_MODAL_HALLUCINATION_PROMPT_VERSION
from aixeval.judge.cross_modal_hallucination import judge_cross_modal_hallucination
from aixeval.judge.evidence import (
    extract_evidence_langfuse,
    extract_tool_calls_langfuse,
    extract_trajectory_langfuse,
)
from aixeval.judge.grounding import PROMPT_VERSION, judge_grounding
from aixeval.judge.pairwise_preference import PROMPT_VERSION as PAIRWISE_PREFERENCE_PROMPT_VERSION
from aixeval.judge.pairwise_preference import judge_pairwise_preference
from aixeval.judge.protected_attribute_bias import (
    DETECTOR_PROMPT_VERSION as PROTECTED_ATTRIBUTE_BIAS_DETECTOR_PROMPT_VERSION,
)
from aixeval.judge.protected_attribute_bias import PROMPT_VERSION as PROTECTED_ATTRIBUTE_BIAS_PROMPT_VERSION
from aixeval.judge.protected_attribute_bias import (
    judge_detect_protected_attribute_signal,
    judge_protected_attribute_bias,
)
from aixeval.judge.provenance import merge_judge_provenance
from aixeval.judge.provider import Provider
from aixeval.judge.refusal import PROMPT_VERSION as REFUSAL_PROMPT_VERSION
from aixeval.judge.refusal import judge_refusal
from aixeval.judge.tool_call_accuracy import PROMPT_VERSION as TOOL_CALL_ACCURACY_PROMPT_VERSION
from aixeval.judge.tool_call_accuracy import judge_tool_call_accuracy
from aixeval.judge.toxicity import PROMPT_VERSION as TOXICITY_PROMPT_VERSION
from aixeval.judge.toxicity import judge_toxicity
from aixeval.judge.trajectory_optimality import PROMPT_VERSION as TRAJECTORY_OPTIMALITY_PROMPT_VERSION
from aixeval.judge.trajectory_optimality import judge_trajectory_optimality


EVIDENCE_EXCERPT_CHARS = 500
# Matches judge_* defaults. Runners used to pass 1024 and starve
# reasoning models (Qwen3 finishes with finish_reason=length and no text).
DEFAULT_JUDGE_MAX_TOKENS = 4000

_KEY_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")
_DEMO_FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "langfuse_gist.json"


class CapturingProvider:
    """Wraps a Provider, stashing the last raw reply for the attachment.

    judge_grounding only returns the parsed score, by design (it has to
    stay a plain ``Callable[[str, str], float]`` to match
    ``score_grounding``'s existing seam). This wrapper is how the
    runner still gets the raw reply for the attachment, without
    changing that contract. Public: also reused by
    ``aixeval.ui.logic`` to surface a live judge's reasoning text
    in the UI, same reason: the ``judge_*`` functions only return a
    parsed score.
    """

    def __init__(self, inner: Provider) -> None:
        self._inner = inner
        self.last_reply: str | None = None

    @property
    def model(self) -> str:
        """Delegate to the wrapped provider's model id."""
        return self._inner.model

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        """Delegate to the wrapped provider, stashing the reply first."""
        reply = self._inner.complete(prompt, max_tokens=max_tokens)
        self.last_reply = reply
        return reply

    def complete_with_image(self, prompt: str, image_path: str, *, max_tokens: int = 1024) -> str:
        """Delegate to the wrapped provider, stashing the reply first."""
        reply = self._inner.complete_with_image(prompt, image_path, max_tokens=max_tokens)
        self.last_reply = reply
        return reply


class _ListCapturingProvider:
    """Wraps a Provider, stashing every raw reply for the attachment.

    ``judge_pairwise_preference`` makes two calls per verdict (position-
    bias correction); ``CapturingProvider``'s single ``last_reply``
    would silently drop the first one. This keeps every reply, in order.
    """

    def __init__(self, inner: Provider) -> None:
        self._inner = inner
        self.replies: list[str] = []

    @property
    def model(self) -> str:
        """Delegate to the wrapped provider's model id."""
        return self._inner.model

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        """Delegate to the wrapped provider, appending the reply first."""
        reply = self._inner.complete(prompt, max_tokens=max_tokens)
        self.replies.append(reply)
        return reply

    def complete_with_image(self, prompt: str, image_path: str, *, max_tokens: int = 1024) -> str:
        """Delegate to the wrapped provider, appending the reply first."""
        reply = self._inner.complete_with_image(prompt, image_path, max_tokens=max_tokens)
        self.replies.append(reply)
        return reply


def run_grounding_judge(
    payload: dict[str, Any],
    record: dict[str, Any],
    client: Provider,
    extract_evidence: Callable[[dict[str, Any]], str | None],
    *,
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Judge one record's grounding against evidence pulled from its raw dump.

    Parameters
    ----------
    payload
        The original dump, already loaded from JSON, not the sealed
        record. Evidence lives here; ingest's copy-only rule already
        stripped it out of ``record``.
    record
        The sealed record; ``output`` is read from it. Never mutated.
    client
        Any ``Provider``.
    extract_evidence
        Which evidence extractor to use for this dump's shape, e.g.
        ``extract_evidence_langfuse``. The caller picks it explicitly;
        no auto-detection, same discipline as ingest.
    max_tokens
        Passed straight to ``judge_grounding``. The default matches
        ``judge_grounding`` (4000). Raise this when evidence is large;
        there is no automatic scaling here.

    Returns
    -------
    tuple[dict, dict]
        ``(new_record, attachment)``. On success, ``new_record`` is a
        copy of ``record`` with ``grounding_judge`` set. On failure, it
        is an unchanged copy. The attachment always has ``status``
        (``"scored"`` or ``"failed"``), ``model``, ``prompt_version``,
        and ``timestamp``; ``evidence_excerpt`` once evidence exists;
        ``reason`` when failed; ``raw_reply`` once a call was made.
    """
    timestamp = datetime.now(UTC).isoformat()
    evidence = extract_evidence(payload)
    if evidence is None:
        return dict(record), _attachment(
            "failed", client, PROMPT_VERSION, timestamp, reason="no evidence extracted from the dump"
        )

    output = record.get("output")
    if not isinstance(output, str) or not output.strip():
        return dict(record), _attachment(
            "failed", client, PROMPT_VERSION, timestamp, evidence=evidence, reason="record has no output to judge"
        )

    wrapped = CapturingProvider(client)
    try:
        score = judge_grounding(evidence, output, wrapped, max_tokens=max_tokens)
    except Exception as exc:  # noqa: BLE001 (bridges four SDKs' distinct exception types into one outcome)
        return dict(record), _attachment(
            "failed",
            client,
            PROMPT_VERSION,
            timestamp,
            evidence=evidence,
            reason=f"judge call failed: {exc}",
            raw_reply=wrapped.last_reply,
        )

    new_record = _record_with_verdict(
        record,
        field="grounding_judge",
        score=score,
        metric_id="grounding",
        client=client,
        prompt_version=PROMPT_VERSION,
    )
    attachment = _attachment(
        "scored", client, PROMPT_VERSION, timestamp, evidence=evidence, raw_reply=wrapped.last_reply
    )
    return new_record, attachment


def run_trajectory_optimality_judge(
    payload: dict[str, Any],
    record: dict[str, Any],
    client: Provider,
    extract_trajectory: Callable[[dict[str, Any]], str | None],
    *,
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Judge one record's trajectory optimality against steps pulled from its raw dump.

    Same shape as ``run_grounding_judge``: needs the raw dump, since
    ingest does not preserve a step trace on the sealed record any more
    than it preserves evidence text.

    Parameters
    ----------
    payload
        The original dump, already loaded from JSON, not the sealed
        record. The step trace lives here.
    record
        The sealed record; ``output`` is read from it. Never mutated.
    client
        Any ``Provider``.
    extract_trajectory
        Which trajectory extractor to use for this dump's shape, e.g.
        ``extract_trajectory_langfuse``. The caller picks it
        explicitly; no auto-detection, same discipline as ingest.
    max_tokens
        Passed straight to ``judge_trajectory_optimality``. See
        ``run_grounding_judge``'s docstring for why this needs real
        headroom for reasoning-capable models.

    Returns
    -------
    tuple[dict, dict]
        ``(new_record, attachment)``. On success, ``new_record`` is a
        copy of ``record`` with ``trajectory_optimality_judge`` set.
        On failure, it is an unchanged copy. The attachment always has
        ``status`` (``"scored"`` or ``"failed"``), ``model``,
        ``prompt_version``, and ``timestamp``; ``trajectory_excerpt``
        once a trajectory exists; ``reason`` when failed; ``raw_reply``
        once a call was made.
    """
    timestamp = datetime.now(UTC).isoformat()
    trajectory = extract_trajectory(payload)
    if trajectory is None:
        return dict(record), _attachment(
            "failed",
            client,
            TRAJECTORY_OPTIMALITY_PROMPT_VERSION,
            timestamp,
            reason="no trajectory extracted from the dump",
        )

    output = record.get("output")
    if not isinstance(output, str) or not output.strip():
        return dict(record), _attachment(
            "failed",
            client,
            TRAJECTORY_OPTIMALITY_PROMPT_VERSION,
            timestamp,
            excerpt=trajectory,
            excerpt_key="trajectory_excerpt",
            reason="record has no output to judge",
        )

    wrapped = CapturingProvider(client)
    try:
        score = judge_trajectory_optimality(trajectory, output, wrapped, max_tokens=max_tokens)
    except Exception as exc:  # noqa: BLE001 (bridges four SDKs' distinct exception types into one outcome)
        return dict(record), _attachment(
            "failed",
            client,
            TRAJECTORY_OPTIMALITY_PROMPT_VERSION,
            timestamp,
            excerpt=trajectory,
            excerpt_key="trajectory_excerpt",
            reason=f"judge call failed: {exc}",
            raw_reply=wrapped.last_reply,
        )

    new_record = _record_with_verdict(
        record,
        field="trajectory_optimality_judge",
        score=score,
        metric_id="trajectory_optimality",
        client=client,
        prompt_version=TRAJECTORY_OPTIMALITY_PROMPT_VERSION,
    )
    attachment = _attachment(
        "scored",
        client,
        TRAJECTORY_OPTIMALITY_PROMPT_VERSION,
        timestamp,
        excerpt=trajectory,
        excerpt_key="trajectory_excerpt",
        raw_reply=wrapped.last_reply,
    )
    return new_record, attachment


def run_abstention_judge(
    record: dict[str, Any],
    client: Provider,
    *,
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Judge one record's abstention from ``output`` alone: no dump needed.

    Unlike ``run_grounding_judge``, there is no evidence to extract:
    the only input the judge needs, ``output``, already lives on the
    sealed record.

    Parameters
    ----------
    record
        The sealed record; ``output`` is read from it. Never mutated.
    client
        Any ``Provider``.
    max_tokens
        Passed straight to ``judge_abstention``. See
        ``run_grounding_judge``'s docstring for why this needs real
        headroom for reasoning-capable models.

    Returns
    -------
    tuple[dict, dict]
        ``(new_record, attachment)``. On success, ``new_record`` is a
        copy of ``record`` with ``abstention_judge`` set. On failure,
        it is an unchanged copy. The attachment always has ``status``
        (``"scored"`` or ``"failed"``), ``model``, ``prompt_version``,
        and ``timestamp``; ``reason`` when failed; ``raw_reply`` once a
        call was made.
    """
    timestamp = datetime.now(UTC).isoformat()
    output = record.get("output")
    if not isinstance(output, str) or not output.strip():
        return dict(record), _attachment(
            "failed", client, ABSTENTION_PROMPT_VERSION, timestamp, reason="record has no output to judge"
        )

    wrapped = CapturingProvider(client)
    try:
        score = judge_abstention(output, wrapped, max_tokens=max_tokens)
    except Exception as exc:  # noqa: BLE001 (bridges four SDKs' distinct exception types into one outcome)
        return dict(record), _attachment(
            "failed",
            client,
            ABSTENTION_PROMPT_VERSION,
            timestamp,
            reason=f"judge call failed: {exc}",
            raw_reply=wrapped.last_reply,
        )

    new_record = _record_with_verdict(
        record,
        field="abstention_judge",
        score=score,
        metric_id="abstention",
        client=client,
        prompt_version=ABSTENTION_PROMPT_VERSION,
    )
    attachment = _attachment("scored", client, ABSTENTION_PROMPT_VERSION, timestamp, raw_reply=wrapped.last_reply)
    return new_record, attachment


def run_refusal_judge(
    record: dict[str, Any],
    client: Provider,
    *,
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Judge one record's refusal from ``output`` alone: no dump needed.

    Same shape as ``run_abstention_judge``: the only input the judge
    needs, ``output``, already lives on the sealed record.

    Parameters
    ----------
    record
        The sealed record; ``output`` is read from it. Never mutated.
    client
        Any ``Provider``.
    max_tokens
        Passed straight to ``judge_refusal``. See
        ``run_grounding_judge``'s docstring for why this needs real
        headroom for reasoning-capable models.

    Returns
    -------
    tuple[dict, dict]
        ``(new_record, attachment)``. On success, ``new_record`` is a
        copy of ``record`` with ``refusal_judge`` set. On failure, it
        is an unchanged copy. The attachment always has ``status``
        (``"scored"`` or ``"failed"``), ``model``, ``prompt_version``,
        and ``timestamp``; ``reason`` when failed; ``raw_reply`` once a
        call was made.
    """
    timestamp = datetime.now(UTC).isoformat()
    output = record.get("output")
    if not isinstance(output, str) or not output.strip():
        return dict(record), _attachment(
            "failed", client, REFUSAL_PROMPT_VERSION, timestamp, reason="record has no output to judge"
        )

    wrapped = CapturingProvider(client)
    try:
        score = judge_refusal(output, wrapped, max_tokens=max_tokens)
    except Exception as exc:  # noqa: BLE001 (bridges four SDKs' distinct exception types into one outcome)
        return dict(record), _attachment(
            "failed",
            client,
            REFUSAL_PROMPT_VERSION,
            timestamp,
            reason=f"judge call failed: {exc}",
            raw_reply=wrapped.last_reply,
        )

    new_record = _record_with_verdict(
        record,
        field="refusal_judge",
        score=score,
        metric_id="refusal",
        client=client,
        prompt_version=REFUSAL_PROMPT_VERSION,
    )
    attachment = _attachment("scored", client, REFUSAL_PROMPT_VERSION, timestamp, raw_reply=wrapped.last_reply)
    return new_record, attachment


def run_protected_attribute_bias_judge(
    record: dict[str, Any],
    client: Provider,
    *,
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Judge one record's bias resistance from its signal and output fields.

    No dump needed: both inputs the judge needs already live on the
    sealed record, same shape as ``run_refusal_judge``/
    ``run_abstention_judge``, just with two text fields instead of one.

    Parameters
    ----------
    record
        The sealed record; ``protected_attribute_signal`` and
        ``output`` are read from it. Never mutated.
    client
        Any ``Provider``.
    max_tokens
        Passed straight to ``judge_protected_attribute_bias``. See
        ``run_grounding_judge``'s docstring for why this needs real
        headroom for reasoning-capable models.

    Returns
    -------
    tuple[dict, dict]
        ``(new_record, attachment)``. On success, ``new_record`` is a
        copy of ``record`` with ``protected_attribute_bias_judge``
        set. On failure, it is an unchanged copy. The attachment
        always has ``status`` (``"scored"`` or ``"failed"``),
        ``model``, ``prompt_version``, and ``timestamp``; ``reason``
        when failed; ``raw_reply`` once a call was made.
    """
    timestamp = datetime.now(UTC).isoformat()
    signal = record.get("protected_attribute_signal")
    output = record.get("output")
    if not isinstance(signal, str) or not signal.strip():
        return dict(record), _attachment(
            "failed",
            client,
            PROTECTED_ATTRIBUTE_BIAS_PROMPT_VERSION,
            timestamp,
            reason="record has no protected_attribute_signal to judge",
        )
    if not isinstance(output, str) or not output.strip():
        return dict(record), _attachment(
            "failed", client, PROTECTED_ATTRIBUTE_BIAS_PROMPT_VERSION, timestamp, reason="record has no output to judge"
        )

    wrapped = CapturingProvider(client)
    try:
        score = judge_protected_attribute_bias(signal, output, wrapped, max_tokens=max_tokens)
    except Exception as exc:  # noqa: BLE001 (bridges four SDKs' distinct exception types into one outcome)
        return dict(record), _attachment(
            "failed",
            client,
            PROTECTED_ATTRIBUTE_BIAS_PROMPT_VERSION,
            timestamp,
            reason=f"judge call failed: {exc}",
            raw_reply=wrapped.last_reply,
        )

    new_record = _record_with_verdict(
        record,
        field="protected_attribute_bias_judge",
        score=score,
        metric_id="protected_attribute_bias",
        client=client,
        prompt_version=PROTECTED_ATTRIBUTE_BIAS_PROMPT_VERSION,
    )
    attachment = _attachment(
        "scored", client, PROTECTED_ATTRIBUTE_BIAS_PROMPT_VERSION, timestamp, raw_reply=wrapped.last_reply
    )
    return new_record, attachment


def run_protected_attribute_bias_detector(
    input_texts: list[str] | tuple[str, ...],
    client: Provider,
    *,
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
) -> tuple[str | None, dict[str, Any]]:
    """Scan a trace's own input text for a candidate protected-attribute signal.

    A detector, not a scorer: same spirit as
    ``aixeval.ingest.audit.audit_expected_in_inputs`` (contamination's
    auto-detect over trace input text) but semantic rather than
    deterministic, so it needs a real judge call and never writes a
    record field itself -- the candidate it returns is always a
    suggestion for a human to confirm or edit, never auto-applied. See
    ``aixeval.ui.logic.merge_bias_signal_into_declared`` for the step
    that turns a confirmed candidate into ``should_resist_bias`` /
    ``protected_attribute_signal`` declared fields.

    Parameters
    ----------
    input_texts
        Raw text pulled from the trace's own spans, typically
        ``aixeval.ingest.trace.collect_trace_input_texts(spans)``.
        Joined into one block before the call; empty/blank entries
        are dropped.
    client
        Any ``Provider``.
    max_tokens
        Passed straight to ``judge_detect_protected_attribute_signal``.
        Default is ``DEFAULT_JUDGE_MAX_TOKENS`` so reasoning models
        (Qwen3) have headroom for hidden thinking before ``SIGNAL:``.

    Returns
    -------
    tuple[str | None, dict]
        ``(candidate_signal, attachment)``. ``candidate_signal`` is
        ``None`` when nothing is found, or when the call fails -- in
        the failure case the attachment's ``status`` is ``"failed"``
        with a ``reason``, otherwise ``"scored"`` whether or not a
        candidate was found (finding nothing is a successful scan,
        not an error).
    """
    timestamp = datetime.now(UTC).isoformat()
    joined = "\n\n".join(text.strip() for text in input_texts if isinstance(text, str) and text.strip())
    if not joined:
        return None, _attachment(
            "failed",
            client,
            PROTECTED_ATTRIBUTE_BIAS_DETECTOR_PROMPT_VERSION,
            timestamp,
            reason="no input text to scan",
        )

    wrapped = CapturingProvider(client)
    try:
        signal = judge_detect_protected_attribute_signal(joined, wrapped, max_tokens=max_tokens)
    except Exception as exc:  # noqa: BLE001 (bridges four SDKs' distinct exception types into one outcome)
        return None, _attachment(
            "failed",
            client,
            PROTECTED_ATTRIBUTE_BIAS_DETECTOR_PROMPT_VERSION,
            timestamp,
            reason=f"detector call failed: {exc}",
            raw_reply=wrapped.last_reply,
        )

    attachment = _attachment(
        "scored", client, PROTECTED_ATTRIBUTE_BIAS_DETECTOR_PROMPT_VERSION, timestamp, raw_reply=wrapped.last_reply
    )
    return signal, attachment


def run_toxicity_judge(
    record: dict[str, Any],
    client: Provider,
    *,
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Judge one record's toxicity from ``output`` alone: no dump needed.

    Same shape as ``run_abstention_judge``/``run_refusal_judge``: the
    only input the judge needs, ``output``, already lives on the
    sealed record. One of two independent mechanisms that write
    ``toxicity_judge``; see ``aixeval.scorecard.toxicity`` for
    the other (classifier-based) path.

    Parameters
    ----------
    record
        The sealed record; ``output`` is read from it. Never mutated.
    client
        Any ``Provider``.
    max_tokens
        Passed straight to ``judge_toxicity``. See
        ``run_grounding_judge``'s docstring for why this needs real
        headroom for reasoning-capable models.

    Returns
    -------
    tuple[dict, dict]
        ``(new_record, attachment)``. On success, ``new_record`` is a
        copy of ``record`` with ``toxicity_judge`` set. On failure, it
        is an unchanged copy. The attachment always has ``status``
        (``"scored"`` or ``"failed"``), ``model``, ``prompt_version``,
        and ``timestamp``; ``reason`` when failed; ``raw_reply`` once a
        call was made.
    """
    timestamp = datetime.now(UTC).isoformat()
    output = record.get("output")
    if not isinstance(output, str) or not output.strip():
        return dict(record), _attachment(
            "failed", client, TOXICITY_PROMPT_VERSION, timestamp, reason="record has no output to judge"
        )

    wrapped = CapturingProvider(client)
    try:
        score = judge_toxicity(output, wrapped, max_tokens=max_tokens)
    except Exception as exc:  # noqa: BLE001 (bridges four SDKs' distinct exception types into one outcome)
        return dict(record), _attachment(
            "failed",
            client,
            TOXICITY_PROMPT_VERSION,
            timestamp,
            reason=f"judge call failed: {exc}",
            raw_reply=wrapped.last_reply,
        )

    new_record = _record_with_verdict(
        record,
        field="toxicity_judge",
        score=score,
        metric_id="toxicity",
        client=client,
        prompt_version=TOXICITY_PROMPT_VERSION,
    )
    attachment = _attachment("scored", client, TOXICITY_PROMPT_VERSION, timestamp, raw_reply=wrapped.last_reply)
    return new_record, attachment


def run_pairwise_preference_judge(
    record: dict[str, Any],
    client: Provider,
    *,
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Judge one record's ``output`` against a declared ``baseline_output``.

    No dump needed: both texts already live on the sealed record,
    same discipline as ``run_abstention_judge``/``run_refusal_judge``.
    If ``expected`` (a gold reference) is present, it is passed to the
    judge as optional context. Unlike every other runner here, the
    judge makes two model calls per verdict, not one: position-bias
    correction, averaging over both presentation orders (see
    ``aixeval.judge.pairwise_preference``).

    Parameters
    ----------
    record
        The sealed record; ``output``, ``baseline_output``, and
        (optionally) ``expected`` are read from it. Never mutated.
    client
        Any ``Provider``.
    max_tokens
        Passed to each of the two underlying judge calls. See
        ``run_grounding_judge``'s docstring for why this needs real
        headroom for reasoning-capable models.

    Returns
    -------
    tuple[dict, dict]
        ``(new_record, attachment)``. On success, ``new_record`` is a
        copy of ``record`` with ``pairwise_preference_judge`` set. On
        failure, it is an unchanged copy. The attachment always has
        ``status`` (``"scored"`` or ``"failed"``), ``model``,
        ``prompt_version``, and ``timestamp``; ``reason`` when failed;
        ``raw_reply`` (both calls' replies, joined) once at least one
        call was made.
    """
    timestamp = datetime.now(UTC).isoformat()
    output = record.get("output")
    if not isinstance(output, str) or not output.strip():
        return dict(record), _attachment(
            "failed", client, PAIRWISE_PREFERENCE_PROMPT_VERSION, timestamp, reason="record has no output to judge"
        )

    baseline = record.get("baseline_output")
    if not isinstance(baseline, str) or not baseline.strip():
        return dict(record), _attachment(
            "failed",
            client,
            PAIRWISE_PREFERENCE_PROMPT_VERSION,
            timestamp,
            reason="record has no baseline_output to compare against",
        )

    reference = record.get("expected")
    reference = reference if isinstance(reference, str) and reference.strip() else None

    wrapped = _ListCapturingProvider(client)
    try:
        score = judge_pairwise_preference(output, baseline, wrapped, reference=reference, max_tokens=max_tokens)
    except Exception as exc:  # noqa: BLE001 (bridges four SDKs' distinct exception types into one outcome)
        return dict(record), _attachment(
            "failed",
            client,
            PAIRWISE_PREFERENCE_PROMPT_VERSION,
            timestamp,
            reason=f"judge call failed: {exc}",
            raw_reply="\n---\n".join(wrapped.replies) or None,
        )

    new_record = _record_with_verdict(
        record,
        field="pairwise_preference_judge",
        score=score,
        metric_id="pairwise_preference",
        client=client,
        prompt_version=PAIRWISE_PREFERENCE_PROMPT_VERSION,
    )
    attachment = _attachment(
        "scored",
        client,
        PAIRWISE_PREFERENCE_PROMPT_VERSION,
        timestamp,
        raw_reply="\n---\n".join(wrapped.replies) or None,
    )
    return new_record, attachment


def run_cross_modal_hallucination_judge(
    record: dict[str, Any],
    client: Provider,
    *,
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Judge one record's cross-modal hallucination from a declared description or path.

    No dump needed: ``image_path``/``image_description`` and
    ``output`` already live on the sealed record, same no-dump-needed
    discipline as ``run_abstention_judge``/``run_refusal_judge``.
    Prefers ``image_path`` (the judge reads the real image) over
    ``image_description`` (text proxy) when a record has both.

    Parameters
    ----------
    record
        The sealed record; ``image_path``/``image_description`` and
        ``output`` are read from it. Never mutated.
    client
        Any ``Provider``.
    max_tokens
        Passed straight to ``judge_cross_modal_hallucination``. See
        ``run_grounding_judge``'s docstring for why this needs real
        headroom for reasoning-capable models.

    Returns
    -------
    tuple[dict, dict]
        ``(new_record, attachment)``. On success, ``new_record`` is a
        copy of ``record`` with ``cross_modal_hallucination_judge``
        set. On failure, it is an unchanged copy. The attachment
        always has ``status`` (``"scored"`` or ``"failed"``),
        ``model``, ``prompt_version``, and ``timestamp``; ``reason``
        when failed; ``raw_reply`` once a call was made.
    """
    timestamp = datetime.now(UTC).isoformat()
    image_path = record.get("image_path")
    image_description = record.get("image_description")
    has_image_path = isinstance(image_path, str) and bool(image_path.strip())
    has_image_description = isinstance(image_description, str) and bool(image_description.strip())
    if not has_image_path and not has_image_description:
        return dict(record), _attachment(
            "failed",
            client,
            CROSS_MODAL_HALLUCINATION_PROMPT_VERSION,
            timestamp,
            reason="record has no image_path or image_description to judge against",
        )

    output = record.get("output")
    if not isinstance(output, str) or not output.strip():
        return dict(record), _attachment(
            "failed",
            client,
            CROSS_MODAL_HALLUCINATION_PROMPT_VERSION,
            timestamp,
            reason="record has no output to judge",
        )

    wrapped = CapturingProvider(client)
    try:
        if has_image_path:
            score = judge_cross_modal_hallucination(output, wrapped, image_path=image_path, max_tokens=max_tokens)
        else:
            score = judge_cross_modal_hallucination(
                output, wrapped, image_description=image_description, max_tokens=max_tokens
            )
    except Exception as exc:  # noqa: BLE001 (bridges four SDKs' distinct exception types into one outcome)
        return dict(record), _attachment(
            "failed",
            client,
            CROSS_MODAL_HALLUCINATION_PROMPT_VERSION,
            timestamp,
            reason=f"judge call failed: {exc}",
            raw_reply=wrapped.last_reply,
        )

    new_record = _record_with_verdict(
        record,
        field="cross_modal_hallucination_judge",
        score=score,
        metric_id="cross_modal_hallucination",
        client=client,
        prompt_version=CROSS_MODAL_HALLUCINATION_PROMPT_VERSION,
    )
    attachment = _attachment(
        "scored", client, CROSS_MODAL_HALLUCINATION_PROMPT_VERSION, timestamp, raw_reply=wrapped.last_reply
    )
    return new_record, attachment


def collect_rerun_scores(
    run_once: Callable[[], tuple[dict[str, Any], dict[str, Any]]],
    field: str,
    *,
    n: int = 3,
) -> list[float]:
    """Call a judge runner ``n`` times on one input, each time collecting one field.

    The missing piece for ``scorecard.test_retest_stability``, which
    only grades an already-assembled ``rerun_scores`` list; this is
    what actually assembles one, by making the repeated calls.

    Parameters
    ----------
    run_once
        A zero-argument callable, already bound to a specific runner
        and input via ``functools.partial``, e.g.
        ``functools.partial(run_grounding_judge, payload, record,
        client, extract_evidence_langfuse)``. Called ``n`` times,
        unchanged: this function does not cache or short-circuit
        repeated identical calls. That would defeat the point of a
        test-retest check: it exists to see whether a live judge call
        actually varies run to run.
    field
        Which field to read off each call's returned record, e.g.
        ``"grounding_judge"``.
    n
        How many times to call ``run_once``. Default 3: the smallest
        number that can show a spread rather than just a pair.

    Returns
    -------
    list[float]
        One score per call that succeeded (``attachment["status"] ==
        "scored"``). A call that fails is silently excluded, not
        counted as a 0.0; the same N/A discipline every other
        metric in this toolkit already follows. Pass the result
        straight to ``scorecard.test_retest_stability``'s
        ``rerun_scores`` field; it treats fewer than 2 scores as N/A
        on its own.
    """
    scores: list[float] = []
    for _ in range(n):
        new_record, attachment = run_once()
        if attachment.get("status") != "scored":
            continue
        value = new_record.get(field)
        if isinstance(value, int | float) and not isinstance(value, bool):
            scores.append(float(value))
    return scores


def run_tool_call_accuracy_check(
    payload: dict[str, Any],
    record: dict[str, Any],
    extract_tool_calls: Callable[[dict[str, Any]], list[dict[str, Any]] | None],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score one record's tool-call accuracy deterministically: no LLM.

    Compares the actual tool calls (extracted from the raw dump)
    against a declared, structured ``gold_tool_calls`` list on the
    sealed record: precision/recall F1 over ``{tool, args}`` pairs,
    the same multiset-overlap idiom ``aixeval.scorecard.f1`` uses
    for text tokens. Reproducible: the same input always gives the
    same score. No tool calls actually made is a legitimate outcome to
    compare, not a failure: it scores 1.0 against an empty gold list
    and 0.0 against a non-empty one, the same as any other multiset
    mismatch.

    Parameters
    ----------
    payload
        The original dump, already loaded from JSON, not the sealed
        record. The actual tool calls live here.
    record
        The sealed record; ``gold_tool_calls`` is read from it. Never
        mutated.
    extract_tool_calls
        Which tool-call extractor to use for this dump's shape, e.g.
        ``extract_tool_calls_langfuse``. The caller picks it
        explicitly; no auto-detection, same discipline as ingest.

    Returns
    -------
    tuple[dict, dict]
        ``(new_record, attachment)``. On success, ``new_record`` is a
        copy of ``record`` with ``tool_call_accuracy_judge`` set,
        the field name both this and ``run_tool_call_accuracy_judge``
        share, see ``aixeval.scorecard.tool_call_accuracy``. On
        failure, it is an unchanged copy. The attachment always has
        ``status`` (``"scored"`` or ``"failed"``) and ``timestamp``,
        no ``model``/``prompt_version``/``raw_reply``, since no LLM is
        called; ``reason`` when failed; ``evidence`` (the specific
        missing/extra calls behind the score, not just the bare F1
        number) when scored with a non-perfect match: unlike an LLM
        judge's rationale, this is exact, not a summary: the full
        mismatch, recoverable from the F1 computation itself rather
        than asked of a model.
    """
    timestamp = datetime.now(UTC).isoformat()
    gold_calls = record.get("gold_tool_calls")
    if not isinstance(gold_calls, list):
        return dict(record), _deterministic_attachment(
            "failed", timestamp, reason="record has no gold_tool_calls to compare against"
        )

    actual_calls = extract_tool_calls(payload) or []
    score = _tool_call_f1(actual_calls, gold_calls)
    missing, extra = _tool_call_mismatches(actual_calls, gold_calls)
    new_record = validate_record({**record, "tool_call_accuracy_judge": score})
    return new_record, _deterministic_attachment("scored", timestamp, evidence=_tool_call_evidence(missing, extra))


def run_tool_call_accuracy_judge(
    payload: dict[str, Any],
    record: dict[str, Any],
    client: Provider,
    extract_trajectory: Callable[[dict[str, Any]], str | None],
    *,
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Judge one record's tool-call accuracy against a declared gold plan.

    Same shape as ``run_trajectory_optimality_judge``: needs the raw
    dump for the trajectory. Also reads ``gold_tool_plan``, a
    one-sentence description of which tools should have been used,
    straight off the sealed record, not the dump.

    Parameters
    ----------
    payload
        The original dump, already loaded from JSON, not the sealed
        record. The step trace lives here.
    record
        The sealed record; ``gold_tool_plan`` and ``output`` are read
        from it. Never mutated.
    client
        Any ``Provider``.
    extract_trajectory
        Which trajectory extractor to use for this dump's shape, e.g.
        ``extract_trajectory_langfuse``.
    max_tokens
        Passed straight to ``judge_tool_call_accuracy``. See
        ``run_grounding_judge``'s docstring for why this needs real
        headroom for reasoning-capable models.

    Returns
    -------
    tuple[dict, dict]
        ``(new_record, attachment)``. On success, ``new_record`` is a
        copy of ``record`` with ``tool_call_accuracy_judge`` set. On
        failure, it is an unchanged copy. The attachment always has
        ``status`` (``"scored"`` or ``"failed"``), ``model``,
        ``prompt_version``, and ``timestamp``; ``trajectory_excerpt``
        once a trajectory exists; ``reason`` when failed; ``raw_reply``
        once a call was made.
    """
    timestamp = datetime.now(UTC).isoformat()
    trajectory = extract_trajectory(payload)
    if trajectory is None:
        return dict(record), _attachment(
            "failed",
            client,
            TOOL_CALL_ACCURACY_PROMPT_VERSION,
            timestamp,
            reason="no trajectory extracted from the dump",
        )

    gold_tool_plan = record.get("gold_tool_plan")
    if not isinstance(gold_tool_plan, str) or not gold_tool_plan.strip():
        return dict(record), _attachment(
            "failed",
            client,
            TOOL_CALL_ACCURACY_PROMPT_VERSION,
            timestamp,
            excerpt=trajectory,
            excerpt_key="trajectory_excerpt",
            reason="record has no gold_tool_plan to judge against",
        )

    output = record.get("output")
    if not isinstance(output, str) or not output.strip():
        return dict(record), _attachment(
            "failed",
            client,
            TOOL_CALL_ACCURACY_PROMPT_VERSION,
            timestamp,
            excerpt=trajectory,
            excerpt_key="trajectory_excerpt",
            reason="record has no output to judge",
        )

    wrapped = CapturingProvider(client)
    try:
        score = judge_tool_call_accuracy(trajectory, gold_tool_plan, output, wrapped, max_tokens=max_tokens)
    except Exception as exc:  # noqa: BLE001 (bridges four SDKs' distinct exception types into one outcome)
        return dict(record), _attachment(
            "failed",
            client,
            TOOL_CALL_ACCURACY_PROMPT_VERSION,
            timestamp,
            excerpt=trajectory,
            excerpt_key="trajectory_excerpt",
            reason=f"judge call failed: {exc}",
            raw_reply=wrapped.last_reply,
        )

    new_record = _record_with_verdict(
        record,
        field="tool_call_accuracy_judge",
        score=score,
        metric_id="tool_call_accuracy",
        client=client,
        prompt_version=TOOL_CALL_ACCURACY_PROMPT_VERSION,
    )
    attachment = _attachment(
        "scored",
        client,
        TOOL_CALL_ACCURACY_PROMPT_VERSION,
        timestamp,
        excerpt=trajectory,
        excerpt_key="trajectory_excerpt",
        raw_reply=wrapped.last_reply,
    )
    return new_record, attachment


def _tool_call_key(call: dict[str, Any]) -> tuple[str, str]:
    """Build a hashable ``(tool, args)`` key; dicts aren't hashable."""
    return str(call.get("tool", "")), json.dumps(call.get("args", {}), sort_keys=True, default=str)


def _tool_call_f1(actual: list[dict[str, Any]], gold: list[dict[str, Any]]) -> float:
    """Score precision/recall F1 over ``{tool, args}`` pairs; ``f1.py``, one tier up.

    Catches both missing required calls (recall) and spurious or wrong
    extra ones (precision), unlike a plain recall-only fraction.
    """
    actual_keys = [_tool_call_key(call) for call in actual]
    gold_keys = [_tool_call_key(call) for call in gold]
    if not actual_keys and not gold_keys:
        return 1.0
    if not actual_keys or not gold_keys:
        return 0.0
    common = Counter(actual_keys) & Counter(gold_keys)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(actual_keys)
    recall = overlap / len(gold_keys)
    return 2 * precision * recall / (precision + recall)


def _tool_call_mismatches(
    actual: list[dict[str, Any]], gold: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(missing, extra)``: gold calls never matched, actual ones never matched.

    Multiset-aware, same counting discipline as ``_tool_call_f1``:
    an extra duplicate of an otherwise-matched call still counts as
    extra. One representative call per unmatched key, not every
    duplicate instance (the key already identifies the shape).
    """
    actual_counter = Counter(_tool_call_key(call) for call in actual)
    gold_counter = Counter(_tool_call_key(call) for call in gold)
    missing_keys = gold_counter - actual_counter
    extra_keys = actual_counter - gold_counter
    gold_by_key = {_tool_call_key(call): call for call in gold}
    actual_by_key = {_tool_call_key(call): call for call in actual}
    missing = [gold_by_key[key] for key in missing_keys]
    extra = [actual_by_key[key] for key in extra_keys]
    return missing, extra


def _describe_tool_call(call: dict[str, Any]) -> str:
    return f"{call.get('tool', '?')}({json.dumps(call.get('args', {}), sort_keys=True, default=str)})"


def _tool_call_evidence(missing: list[dict[str, Any]], extra: list[dict[str, Any]]) -> str | None:
    """Give a human-readable account of exactly which calls the F1 score penalizes.

    ``None`` on a perfect match: nothing to explain. Exact, not a
    summary: every mismatch named, not an LLM's paraphrase of them.
    """
    if not missing and not extra:
        return None
    parts = []
    if missing:
        parts.append("missing gold call(s): " + ", ".join(_describe_tool_call(call) for call in missing))
    if extra:
        parts.append("extra/unexpected call(s): " + ", ".join(_describe_tool_call(call) for call in extra))
    return "; ".join(parts)


def _record_with_verdict(
    record: dict[str, Any],
    *,
    field: str,
    score: float,
    metric_id: str,
    client: Provider,
    prompt_version: str,
) -> dict[str, Any]:
    """Seal a successful verdict plus its provenance onto a copy of ``record``."""
    return validate_record(merge_judge_provenance({**record, field: score}, metric_id, client, prompt_version))


def _deterministic_attachment(
    status: str, timestamp: str, *, reason: str | None = None, evidence: str | None = None
) -> dict[str, Any]:
    """Like ``_attachment``, but for a path with no ``client``: no LLM was called."""
    attachment: dict[str, Any] = {"status": status, "timestamp": timestamp}
    if reason is not None:
        attachment["reason"] = reason
    if evidence is not None:
        attachment["evidence"] = evidence
    return attachment


def _attachment(
    status: str,
    client: Provider,
    prompt_version: str,
    timestamp: str,
    *,
    evidence: str | None = None,
    excerpt: str | None = None,
    excerpt_key: str = "evidence_excerpt",
    reason: str | None = None,
    raw_reply: str | None = None,
) -> dict[str, Any]:
    attachment: dict[str, Any] = {
        "status": status,
        "model": client.model,
        "prompt_version": prompt_version,
        "timestamp": timestamp,
    }
    if evidence is not None:
        attachment["evidence_excerpt"] = _excerpt(evidence)
    if excerpt is not None:
        attachment[excerpt_key] = _excerpt(excerpt)
    if reason is not None:
        attachment["reason"] = reason
    if raw_reply is not None:
        attachment["raw_reply"] = raw_reply
    return attachment


def _excerpt(evidence: str) -> str:
    if len(evidence) <= EVIDENCE_EXCERPT_CHARS:
        return evidence
    return evidence[:EVIDENCE_EXCERPT_CHARS] + "…"


def print_runner_example() -> None:
    """Run the full pipeline live on one fixture, if a provider key is set.

    ingest_langfuse -> sealed record; run_grounding_judge -> verdict +
    attachment. Same fixture the integration test uses.
    """
    load_dotenv()
    if not any(os.environ.get(var) for var in _KEY_ENV_VARS):
        print("Set a provider API key (see .env.example) to run a live call. Skipping.")
        return
    record = ingest_langfuse(_DEMO_FIXTURE)
    payload = load_json_object(_DEMO_FIXTURE)
    client = build_client_from_env()
    new_record, attachment = run_grounding_judge(payload, record, client, extract_evidence_langfuse)
    print(f"provider:        {attachment['model']}")
    print(f"status:          {attachment['status']}")
    print(f"grounding_judge: {new_record.get('grounding_judge')}")
    print(f"reason:          {attachment.get('reason')}")


_TOOL_CALL_DEMO_PAYLOAD: dict[str, Any] = {
    "observations": [
        {"type": "SPAN", "name": "agent", "input": "weather in Paris in F", "output": "planning"},
        {"type": "TOOL", "name": "get_weather", "input": {"city": "Paris"}, "output": {"temp_c": 20}},
        {"type": "GENERATION", "name": "llm", "output": "It's about 68F in Paris."},
    ]
}


def print_tool_call_accuracy_runner_example() -> None:
    """Run both tool_call_accuracy mechanisms live on the same scenario.

    Deterministic (``run_tool_call_accuracy_check``) needs no provider
    key. Judge-based (``run_tool_call_accuracy_judge``) does. Same
    weather/Paris scenario both ways: gold expects ``get_weather`` then
    ``convert_temperature``, but only ``get_weather`` was actually
    called; both mechanisms should flag the gap, not necessarily
    agree on the exact number (different mechanisms, same field).
    """
    output = "It's about 68F in Paris."
    deterministic_record = validate_record(
        {
            "output": output,
            "gold_tool_calls": [
                {"tool": "get_weather", "args": {"city": "Paris"}},
                {"tool": "convert_temperature", "args": {"from": "C", "to": "F"}},
            ],
        }
    )
    new_record, attachment = run_tool_call_accuracy_check(
        _TOOL_CALL_DEMO_PAYLOAD, deterministic_record, extract_tool_calls_langfuse
    )
    print("Deterministic path (no LLM):")
    print(f"  tool_call_accuracy_judge: {new_record.get('tool_call_accuracy_judge')}")
    print(f"  attachment:               {attachment}")
    print()

    if not any(os.environ.get(var) for var in _KEY_ENV_VARS):
        print("Judge path: set a provider API key (see .env.example) to run a live call. Skipping.")
        return
    judge_record = validate_record(
        {
            "output": output,
            "gold_tool_plan": (
                "Should call get_weather for the city, then convert_temperature from Celsius to Fahrenheit."
            ),
        }
    )
    client = build_client_from_env()
    new_record, attachment = run_tool_call_accuracy_judge(
        _TOOL_CALL_DEMO_PAYLOAD, judge_record, client, extract_trajectory_langfuse, max_tokens=4000
    )
    print("Judge path (LLM):")
    print(f"  provider:                 {attachment.get('model')}")
    print(f"  status:                   {attachment['status']}")
    print(f"  tool_call_accuracy_judge: {new_record.get('tool_call_accuracy_judge')}")
    print(f"  reason:                   {attachment.get('reason')}")


if __name__ == "__main__":
    print_runner_example()
    print()
    print_tool_call_accuracy_runner_example()
