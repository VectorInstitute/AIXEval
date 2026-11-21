"""Pure orchestration for the Streamlit app: no Streamlit import, unit-testable.

Live-judge dispatch lives in ``aixeval.judge.live``. This module scores
a chosen metric subset, explains rows, and derives run facts for the
metric suggester.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from aixeval.ingest.pipeline import IngestResult
from aixeval.ingest.trace import Span, derive_agents
from aixeval.ingest.trace_taubench import extract_tool_calls_taubench
from aixeval.judge.evidence import extract_tool_calls_langfuse
from aixeval.judge.hard_stop_analysis import analyze_hard_stops
from aixeval.judge.live import LIVE_JUDGE_METRICS
from aixeval.judge.provenance import models_from_provenance
from aixeval.judge.provider import Provider
from aixeval.judge.runner import run_tool_call_accuracy_check
from aixeval.scorecard.definitions import definition_for_any_metric, definition_for_na_reason
from aixeval.scorecard.metrics import score_metrics
from aixeval.scorecard.toxicity import score_toxicity


# LIVE_JUDGE_METRICS is defined in aixeval.judge.live. This module only
# needs it for replay captions; the CLI imports dispatch from judge.live.

# bertscore/meteor/clip_score have real definitions (OPT_IN_METRIC_DEFINITIONS)
# but are deliberately never wired into score_selected: a quick,
# interactive upload-and-score UI is the wrong place for bertscore's/
# clip_score's blocking multi-GB first-use model download (no progress
# indicator in a Streamlit request thread) or meteor's NLTK corpus
# download plus its separate, already-documented short-text scoring
# quirk (it penalizes short correct answers; see meteor.py). Decided,
# not pending: these three stay CLI/notebook-only tools. The app
# disables their checkboxes rather than letting a user pick one and
# hit a multi-minute hang or a misleading grade.
UI_DISABLED_OPT_IN: frozenset[str] = frozenset({"bertscore", "meteor", "clip_score"})


def merge_bias_signal_into_declared(declared: dict[str, Any] | None, signal: str) -> dict[str, Any]:
    """Turn a confirmed bias-detector candidate into declared fields.

    The detector (``run_protected_attribute_bias_detector``) only ever
    returns a *candidate*: this is the one place that candidate becomes
    the declared ``should_resist_bias``/``protected_attribute_signal``
    pair ``score_protected_attribute_bias`` reads, and it only runs
    after a human has looked at the candidate (edited or not) and
    chosen to apply it -- never automatically from the detector's own
    output. Existing declared fields are kept; these two keys are set
    or overwritten.
    """
    merged = dict(declared) if declared else {}
    merged["should_resist_bias"] = True
    merged["protected_attribute_signal"] = signal
    return merged


def tool_call_extractor_for(payload: dict[str, Any]) -> Callable[[dict[str, Any]], list[dict[str, Any]] | None] | None:
    """Pick the deterministic tool-call extractor for this payload's shape, if any.

    Only Langfuse and tau-bench traces have a shipped navigator
    (``extract_tool_calls_langfuse``/``extract_tool_calls_taubench``) --
    OTel and Patronus don't yet, same asymmetry ``ingest_any`` already
    has for its own Langfuse-only metadata bridge. Returns ``None``
    rather than guessing a shape-specific walk for an unsupported format.
    """
    if isinstance(payload.get("observations"), list):
        return extract_tool_calls_langfuse
    if isinstance(payload.get("traj"), list):
        return extract_tool_calls_taubench
    return None


def run_deterministic_tool_call_check(
    payload: dict[str, Any], record: dict[str, Any]
) -> tuple[dict[str, Any], str | None]:
    """Run the no-LLM ``tool_call_accuracy`` check if this payload's shape supports it.

    Returns ``(record, None)`` unchanged if no extractor exists for
    this payload's shape, or if the record has no declared
    ``gold_tool_calls`` to compare against; both are the ordinary N/A
    path ``score_tool_call_accuracy`` already reports on its own, not
    an error this function needs to surface. The second element is the
    attachment's ``evidence`` (which calls were missing/extra) when the
    check actually scored and the match wasn't perfect: for the UI to
    show next to the raw score, same place a live judge's rationale goes.
    """
    extractor = tool_call_extractor_for(payload)
    if extractor is None:
        return record, None
    new_record, attachment = run_tool_call_accuracy_check(payload, record, extractor)
    return new_record, attachment.get("evidence")


def score_all(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Score every metric this UI can show, without calling any model.

    ``score_metrics()`` computes every auto-wired metric (cheap,
    deterministic). ``toxicity`` is the one opt-in metric wired into
    this UI, scored only from a ``toxicity_judge`` verdict already on
    the record. Without one its row is N/A (``missing_evidence``):
    ``score_toxicity`` would otherwise fall back to loading the local
    ``unitary/toxic-bert`` classifier, the same blocking model load
    ``UI_DISABLED_OPT_IN`` keeps out of this UI.
    """
    rows = score_metrics(record)
    if record.get("toxicity_judge") is not None:
        rows.append(score_toxicity(record))
    else:
        rows.append({"metric_id": "toxicity", "status": "na", "na_reason": "missing_evidence"})
    return rows


def score_selected(record: dict[str, Any], selected_ids: set[str]) -> list[dict[str, Any]]:
    """Score ``record`` and keep only the rows the user actually selected."""
    return [row for row in score_all(record) if row["metric_id"] in selected_ids]


def run_hard_stop_analysis(
    explanation: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    record: dict[str, Any],
    spans: list[Span] | None,
    rationale: dict[str, str],
    client: Provider,
) -> tuple[dict[str, Any] | None, str | None]:
    """Run the incident-analysis call when a hard stop fired.

    Returns ``(analysis, error)``. Both are ``None`` when nothing fired.
    Any failure (provider error, malformed reply, a cited step that is
    not in the trace) comes back as ``error`` instead of raising, so the
    card itself always renders.
    """
    if not explanation["hard_stops"]:
        return None, None
    output = record.get("output")
    try:
        analysis = analyze_hard_stops(
            explanation,
            rows,
            output=output if isinstance(output, str) else None,
            spans=spans,
            rationale=rationale,
            client=client,
        )
    except Exception as exc:  # noqa: BLE001 (an explanation failure must never block the card)
        return None, f"{type(exc).__name__}: {exc}"
    return analysis, None


def analysis_matches(analysis: dict[str, Any] | None, explanation: dict[str, Any]) -> bool:
    """Whether a saved analysis covers exactly the hard stops that fired this time."""
    if not analysis:
        return False
    fired = [stop["hard_stop"] for stop in explanation["hard_stops"]]
    return bool(fired) and list(analysis.get("hard_stops", [])) == fired


SAVED_VERDICTS_LABEL = "saved scenario verdicts (judge model not recorded)"


def judge_label(provider: str, model: str) -> str:
    """One short, human-readable name for a judge: ``"vllm · gemma-3-12b-it"``."""
    return f"{provider} · {model}"


def replays_judge_verdicts(replay_record: dict[str, Any] | None, selected_ids: set[str]) -> bool:
    """Return whether any selected judge verdict comes from a saved record."""
    if not replay_record:
        return False
    return any(LIVE_JUDGE_METRICS[m].record_field in replay_record for m in selected_ids & LIVE_JUDGE_METRICS.keys())


def describe_judges(
    *, live_judge: str | None, replayed: bool, replay_record: dict[str, Any] | None = None
) -> str | None:
    """Name the judge(s) that produced a run's verdicts.

    ``None`` when no judge was involved at all (every selected metric
    is deterministic), so the UI shows nothing rather than a made-up name.
    Replayed verdicts use ``judge_provenance`` on ``replay_record``
    when present; older saved runs without it keep
    ``SAVED_VERDICTS_LABEL``.
    """
    saved = None
    if replayed:
        models = models_from_provenance(replay_record)
        saved = ", ".join(models) if models else SAVED_VERDICTS_LABEL
    parts = [label for label in (live_judge, saved) if label]
    return " + ".join(parts) or None


def span_rows(spans: list[Span]) -> list[dict[str, Any]]:
    """Flatten a span tree into display rows, parents before their children.

    Siblings keep their order in ``spans`` (that order is step order for
    ``trajectory_optimality``, so the viewer must not re-sort it). A span
    whose ``parent_id`` points at no span in the list is shown as a root
    rather than dropped. ``depth`` is 0 for a root.
    """
    ids = {span.id for span in spans}
    children: dict[str | None, list[Span]] = {}
    for span in spans:
        parent = span.parent_id if span.parent_id in ids else None
        children.setdefault(parent, []).append(span)

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def visit(span: Span, depth: int) -> None:
        if span.id in seen:
            return
        seen.add(span.id)
        rows.append(
            {
                "depth": depth,
                "id": span.id,
                "kind": span.kind,
                "agent": span.agent,
                "name": span.source_type,
                "status": span.status,
                "start": span.start,
                "end": span.end,
                "duration_s": _duration_seconds(span.start, span.end),
                "input": span.input,
                "output": span.output,
            }
        )
        for child in children.get(span.id, []):
            visit(child, depth + 1)

    for root in children.get(None, []):
        visit(root, 0)
    return rows


def _duration_seconds(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    try:
        begin = datetime.fromisoformat(start.replace("Z", "+00:00"))
        finish = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (finish - begin).total_seconds()


def compare_rows(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Line up two runs' metric rows by ``metric_id`` for a side-by-side table.

    Keeps every metric either run scored; a metric N/A in both is
    dropped (nothing to compare). ``delta`` is ``grade_b - grade_a`` when
    both scored, else ``None``.
    """
    by_id_a = {row["metric_id"]: row for row in rows_a}
    by_id_b = {row["metric_id"]: row for row in rows_b}
    lined_up = []
    for metric_id in sorted(set(by_id_a) | set(by_id_b)):
        grade_a = _grade_or_none(by_id_a.get(metric_id))
        grade_b = _grade_or_none(by_id_b.get(metric_id))
        if grade_a is None and grade_b is None:
            continue
        delta = grade_b - grade_a if grade_a is not None and grade_b is not None else None
        lined_up.append({"metric_id": metric_id, "grade_a": grade_a, "grade_b": grade_b, "delta": delta})
    return lined_up


def _grade_or_none(row: dict[str, Any] | None) -> int | None:
    if row is None or row["status"] != "scored":
        return None
    grade: int = row["grade"]
    return grade


def explain_row(row: dict[str, Any], rationale_by_metric: dict[str, str]) -> str:
    """One human-readable sentence (or two) for a single scored/N/A row.

    ``rationale_by_metric``'s text isn't always a judge's prose: for a
    deterministic metric like ``tool_call_accuracy`` (see
    ``run_deterministic_tool_call_check``) it's an exact mismatch
    breakdown, not a model's summary, so the label here is the
    neutral "Detail", not "Judge's reasoning".
    """
    definition = definition_for_any_metric(row["metric_id"])
    if row["status"] == "scored":
        base = f"{definition} Raw score {row['raw_score']:.3f} -> grade {row['grade']}/4."
        rationale = rationale_by_metric.get(row["metric_id"])
        return f"{base} Detail: {rationale}" if rationale else base
    return f"{definition} N/A: {definition_for_na_reason(row['na_reason'])}"


def derive_run_facts(result: IngestResult, *, image_path: str | None) -> dict[str, Any]:
    """Objective facts about one ingested run, for the metric suggester.

    Checked directly against ``result``'s own spans/trajectory, never
    re-derived from anything the user typed: the whole point is giving
    the suggester something it doesn't have to trust a free-text
    description for. See ``metric_suggester.suggest_relevant_metrics``'s
    own docstring for why this matters (a forgotten "it's multi-agent"
    or "there's an image" in the description used to go unnoticed).

    Returns a dict of keyword arguments ready to splat into
    ``suggest_relevant_metrics(**derive_run_facts(result, image_path=...))``.
    """
    return {
        "source_kind": result.source_kind,
        "has_trajectory": result.trajectory is not None,
        "num_distinct_agents": len(derive_agents(result.spans)) if result.spans is not None else None,
        "has_image": image_path is not None,
    }
