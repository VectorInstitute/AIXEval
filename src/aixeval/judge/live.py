"""Shared live-judge dispatch for the CLI and the UI.

Calls ``judge_*`` for a selected metric set. Does not import Streamlit
or ``aixeval.ui``. Grounding uses dump evidence (same extractors as
``run_grounding_judge``), not ingest trajectory text.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from aixeval.ingest.trace import Span, collect_trace_input_texts
from aixeval.judge.abstention import PROMPT_VERSION as ABSTENTION_PROMPT_VERSION
from aixeval.judge.abstention import judge_abstention
from aixeval.judge.cross_modal_hallucination import PROMPT_VERSION as CROSS_MODAL_HALLUCINATION_PROMPT_VERSION
from aixeval.judge.cross_modal_hallucination import judge_cross_modal_hallucination
from aixeval.judge.evidence import extract_evidence_langfuse
from aixeval.judge.grounding import PROMPT_VERSION as GROUNDING_PROMPT_VERSION
from aixeval.judge.grounding import judge_grounding
from aixeval.judge.pairwise_preference import PROMPT_VERSION as PAIRWISE_PREFERENCE_PROMPT_VERSION
from aixeval.judge.pairwise_preference import judge_pairwise_preference
from aixeval.judge.protected_attribute_bias import PROMPT_VERSION as PROTECTED_ATTRIBUTE_BIAS_PROMPT_VERSION
from aixeval.judge.protected_attribute_bias import judge_protected_attribute_bias
from aixeval.judge.provenance import merge_judge_provenance
from aixeval.judge.provider import Provider
from aixeval.judge.refusal import PROMPT_VERSION as REFUSAL_PROMPT_VERSION
from aixeval.judge.refusal import judge_refusal
from aixeval.judge.runner import CapturingProvider
from aixeval.judge.tool_call_accuracy import PROMPT_VERSION as TOOL_CALL_ACCURACY_PROMPT_VERSION
from aixeval.judge.tool_call_accuracy import judge_tool_call_accuracy
from aixeval.judge.toxicity import PROMPT_VERSION as TOXICITY_PROMPT_VERSION
from aixeval.judge.toxicity import judge_toxicity
from aixeval.judge.trajectory_optimality import PROMPT_VERSION as TRAJECTORY_OPTIMALITY_PROMPT_VERSION
from aixeval.judge.trajectory_optimality import judge_trajectory_optimality


@dataclass(frozen=True)
class _JudgeCallArgs:
    """Everything a single live-judge call might need from the running record."""

    record: dict[str, Any]
    output: str
    trajectory: str | None
    evidence: str | None
    image_path: str | None
    capturing: CapturingProvider


@dataclass(frozen=True)
class LiveJudgeSpec:
    """One live-judge metric and what it needs.

    ``run`` returns ``None`` (not an error) when a precondition beyond
    ``output`` isn't met (e.g. no trajectory, no declared
    ``baseline_output``), the same "silently skipped, scorer reports
    N/A later" discipline ``run_live_judges`` already used for
    trajectory/image gating before this became per-metric. ``possible``
    is the exact same precondition, factored out so a caller can ask
    "would this actually run right now" without spending a judge call.
    """

    record_field: str
    run: Callable[[_JudgeCallArgs], float | None]
    possible: Callable[[_JudgeCallArgs], bool]
    prompt_version: str


def _has_trajectory(ctx: _JudgeCallArgs) -> bool:
    return bool(ctx.trajectory)


def _has_evidence(ctx: _JudgeCallArgs) -> bool:
    return bool(ctx.evidence)


def _has_image(ctx: _JudgeCallArgs) -> bool:
    return bool(ctx.image_path)


def _always_possible(ctx: _JudgeCallArgs) -> bool:  # noqa: ARG001 (uniform Callable[[_JudgeCallArgs], bool] shape)
    return True


def _has_gold_tool_plan(ctx: _JudgeCallArgs) -> bool:
    if not _has_trajectory(ctx):
        return False
    gold_tool_plan = ctx.record.get("gold_tool_plan")
    return isinstance(gold_tool_plan, str) and bool(gold_tool_plan.strip())


def _has_baseline_output(ctx: _JudgeCallArgs) -> bool:
    baseline = ctx.record.get("baseline_output")
    return isinstance(baseline, str) and bool(baseline.strip())


def _has_protected_attribute_signal(ctx: _JudgeCallArgs) -> bool:
    signal = ctx.record.get("protected_attribute_signal")
    return isinstance(signal, str) and bool(signal.strip())


def _run_trajectory_optimality(ctx: _JudgeCallArgs) -> float | None:
    # Inline check (not _has_trajectory) so mypy narrows ctx.trajectory to
    # str for the call below; narrowing doesn't cross a function call.
    if not ctx.trajectory:
        return None
    return judge_trajectory_optimality(ctx.trajectory, ctx.output, ctx.capturing)


def _run_grounding(ctx: _JudgeCallArgs) -> float | None:
    if not ctx.evidence:
        return None
    return judge_grounding(ctx.evidence, ctx.output, ctx.capturing)


def _run_toxicity(ctx: _JudgeCallArgs) -> float:
    return judge_toxicity(ctx.output, ctx.capturing)


def _run_cross_modal_hallucination(ctx: _JudgeCallArgs) -> float | None:
    if not _has_image(ctx):
        return None
    return judge_cross_modal_hallucination(ctx.output, ctx.capturing, image_path=ctx.image_path)


def _run_abstention(ctx: _JudgeCallArgs) -> float:
    return judge_abstention(ctx.output, ctx.capturing)


def _run_refusal(ctx: _JudgeCallArgs) -> float:
    return judge_refusal(ctx.output, ctx.capturing)


def _run_protected_attribute_bias(ctx: _JudgeCallArgs) -> float | None:
    if not _has_protected_attribute_signal(ctx):
        return None
    signal = ctx.record["protected_attribute_signal"]
    return judge_protected_attribute_bias(signal, ctx.output, ctx.capturing)


def _run_tool_call_accuracy(ctx: _JudgeCallArgs) -> float | None:
    if not ctx.trajectory:
        return None
    gold_tool_plan = ctx.record.get("gold_tool_plan")
    if not isinstance(gold_tool_plan, str) or not gold_tool_plan.strip():
        return None
    return judge_tool_call_accuracy(ctx.trajectory, gold_tool_plan, ctx.output, ctx.capturing)


def _run_pairwise_preference(ctx: _JudgeCallArgs) -> float | None:
    if not _has_baseline_output(ctx):
        return None
    baseline = ctx.record["baseline_output"]
    reference = ctx.record.get("expected")
    reference = reference if isinstance(reference, str) and reference.strip() else None
    # Two judge calls under the hood (position-bias correction); CapturingProvider
    # only keeps the last reply, so the rationale shown is the second call's only,
    # a deliberate simplification, not a bug (see judge/pairwise_preference.py).
    return judge_pairwise_preference(ctx.output, baseline, ctx.capturing, reference=reference)


LIVE_JUDGE_METRICS: dict[str, LiveJudgeSpec] = {
    "trajectory_optimality": LiveJudgeSpec(
        "trajectory_optimality_judge", _run_trajectory_optimality, _has_trajectory, TRAJECTORY_OPTIMALITY_PROMPT_VERSION
    ),
    "grounding": LiveJudgeSpec("grounding_judge", _run_grounding, _has_evidence, GROUNDING_PROMPT_VERSION),
    "toxicity": LiveJudgeSpec("toxicity_judge", _run_toxicity, _always_possible, TOXICITY_PROMPT_VERSION),
    "cross_modal_hallucination": LiveJudgeSpec(
        "cross_modal_hallucination_judge",
        _run_cross_modal_hallucination,
        _has_image,
        CROSS_MODAL_HALLUCINATION_PROMPT_VERSION,
    ),
    "abstention": LiveJudgeSpec("abstention_judge", _run_abstention, _always_possible, ABSTENTION_PROMPT_VERSION),
    "refusal": LiveJudgeSpec("refusal_judge", _run_refusal, _always_possible, REFUSAL_PROMPT_VERSION),
    "tool_call_accuracy": LiveJudgeSpec(
        "tool_call_accuracy_judge", _run_tool_call_accuracy, _has_gold_tool_plan, TOOL_CALL_ACCURACY_PROMPT_VERSION
    ),
    "pairwise_preference": LiveJudgeSpec(
        "pairwise_preference_judge", _run_pairwise_preference, _has_baseline_output, PAIRWISE_PREFERENCE_PROMPT_VERSION
    ),
    "protected_attribute_bias": LiveJudgeSpec(
        "protected_attribute_bias_judge",
        _run_protected_attribute_bias,
        _has_protected_attribute_signal,
        PROTECTED_ATTRIBUTE_BIAS_PROMPT_VERSION,
    ),
}


def extract_live_grounding_evidence(
    payload: dict[str, Any] | None = None,
    spans: list[Span] | None = None,
) -> str | None:
    """Return the evidence text live grounding should judge against.

    Langfuse-shaped dumps use ``extract_evidence_langfuse``, the same
    extractor ``run_grounding_judge`` takes. Other traces fall back to
    span input texts (``collect_trace_input_texts``): what the agent
    saw, not the step log ``format_trajectory`` builds. Trajectory
    text is never used. ``None`` when neither source has usable input.
    """
    if payload is not None and isinstance(payload.get("observations"), list):
        from_dump = extract_evidence_langfuse(payload)
        if from_dump:
            return from_dump
    if spans:
        texts = collect_trace_input_texts(spans)
        return "\n\n".join(texts) if texts else None
    return None


def run_live_judges(
    record: dict[str, Any],
    *,
    trajectory: str | None,
    image_path: str | None,
    selected_ids: set[str],
    client: Provider,
    evidence: str | None = None,
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    """Run a live judge call for each selected metric this dispatcher supports.

    Returns ``(updated_record, rationale_by_metric, error_by_metric)``.
    ``updated_record`` is a copy of ``record`` with each successful
    verdict's field set. A metric that needed a trajectory, evidence,
    or image that isn't available is silently skipped (the later
    scorer call reports it N/A, same as any other missing-evidence
    case); that is not an error. A judge call that raises anything (a
    bad reply, or a provider SDK error such as a rate limit, auth
    failure or timeout, none of which subclass ``ValueError``/
    ``OSError``) is caught per metric so one failing judge never
    blocks the others; its message lands in ``error_by_metric``
    instead of crashing the run.

    A metric whose ``record_field`` is already on ``record`` is not
    re-judged: a saved verdict (a replayed scenario) is reused rather
    than spending another call. Each new verdict also writes
    ``judge_provenance`` for that metric (model + prompt version).
    """
    updated = dict(record)
    rationale: dict[str, str] = {}
    errors: dict[str, str] = {}
    capturing = CapturingProvider(client)

    for metric_id in selected_ids & LIVE_JUDGE_METRICS.keys():
        spec = LIVE_JUDGE_METRICS[metric_id]
        if spec.record_field in updated:
            continue
        output = updated.get("output")
        if not isinstance(output, str) or not output.strip():
            continue
        ctx = _JudgeCallArgs(
            record=updated,
            output=output,
            trajectory=trajectory,
            evidence=evidence,
            image_path=image_path,
            capturing=capturing,
        )
        try:
            verdict = spec.run(ctx)
        except Exception as exc:  # noqa: BLE001 (any provider failure is isolated to this one metric)
            errors[metric_id] = f"{type(exc).__name__}: {exc}"
            continue
        if verdict is None:
            continue
        updated[spec.record_field] = verdict
        updated = merge_judge_provenance(updated, metric_id, client, spec.prompt_version)
        if capturing.last_reply:
            rationale[metric_id] = capturing.last_reply.strip()[:500]
    return updated, rationale, errors


class _UnreachableProvider:
    """A ``Provider`` that must never actually be called.

    ``_JudgeCallArgs.capturing`` is typed as a real ``CapturingProvider``,
    which has to wrap *some* ``Provider``; but ``live_judge_is_possible``
    only ever calls a spec's ``possible`` predicate, which (by every
    predicate's own contract) reads ``ctx.record``/``trajectory``/
    ``evidence``/``image_path`` only, never ``ctx.capturing``. This
    stands in so no real client/API key is needed just to check a
    precondition.
    """

    @property
    def model(self) -> str:
        return "unreachable"

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:  # noqa: ARG002
        msg = "live_judge_is_possible should never actually call the provider"
        raise AssertionError(msg)

    def complete_with_image(self, prompt: str, image_path: str, *, max_tokens: int = 1024) -> str:  # noqa: ARG002
        msg = "live_judge_is_possible should never actually call the provider"
        raise AssertionError(msg)


def live_judge_is_possible(
    metric_id: str,
    record: dict[str, Any],
    *,
    trajectory: str | None,
    image_path: str | None,
    evidence: str | None = None,
) -> bool:
    """Report whether ``run_live_judges`` would actually attempt this metric now.

    Checks the exact same preconditions ``run_live_judges`` itself uses
    (output present, plus the metric's own ``LiveJudgeSpec.possible``) --
    without making a judge call or needing a real client/API key.
    """
    spec = LIVE_JUDGE_METRICS.get(metric_id)
    if spec is None:
        return False
    output = record.get("output")
    if not isinstance(output, str) or not output.strip():
        return False
    ctx = _JudgeCallArgs(
        record=record,
        output=output,
        trajectory=trajectory,
        evidence=evidence,
        image_path=image_path,
        capturing=CapturingProvider(_UnreachableProvider()),
    )
    return spec.possible(ctx)
