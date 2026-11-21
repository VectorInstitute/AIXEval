"""One entrypoint for any supported dump shape: detect, parse, bridge to a record.

The multi-agent trace parsers (Langfuse/OTel/Patronus, ``ingest.trace``)
and the simple single-record dump path (``ingest.dumps``) produce two
different shapes: a ``Span`` list versus a flat sealed-record dict.
Most metrics need declared fields (gold answer, budgets, safety flags)
that no raw trace or dump carries on its own. This module bridges both
paths into one call a UI makes once, and is explicit about what it can
and cannot auto-derive: it never guesses a gold field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aixeval.ingest.audit import audit_expected_in_inputs
from aixeval.ingest.dumps import simple_record_fields
from aixeval.ingest.langfuse import langfuse_record_fields
from aixeval.ingest.overlay import build_overlay_fields
from aixeval.ingest.record import validate_record
from aixeval.ingest.trace import Span, collect_trace_input_texts, detect_and_parse, format_trajectory
from aixeval.ingest.trace_inspect import parse_inspect_sample, read_inspect_eval


@dataclass
class IngestResult:
    """One ingested run: a scoreable record, plus the raw trace if it was one.

    ``spans``/``trajectory`` are ``None`` for a simple (non-trace) dump:
    there is no step-by-step trail to show a judge or a MAS metric.
    ``contamination_evidence`` is ``None`` unless ``audit_contamination``
    found a hit, kept off the sealed record itself, same reason a live
    judge's rationale isn't written into the record either.
    """

    record: dict[str, Any]
    spans: list[Span] | None
    trajectory: str | None
    source_kind: str  # "trace" or "simple_dump"
    contamination_evidence: str | None = None


def ingest_any(
    payload: dict[str, Any],
    *,
    declared_fields: dict[str, Any] | None = None,
    overlay: dict[str, Any] | None = None,
    audit_contamination: bool = False,
) -> IngestResult:
    """Ingest one dump, whatever its shape, and merge in any user-declared fields.

    Tries the multi-agent trace parsers first (``detect_and_parse``:
    Langfuse/OTel/Patronus). If the payload doesn't match any of those
    shapes, falls back to the simple single-record dump path.

    For a Langfuse-shaped trace, also runs the existing
    ``ingest_langfuse`` field rules (``expected`` from ``metadata``,
    ``stated_confidence``, ``latency_sec``, last-generation ``output``)
    on top of the trace; that extraction only exists for Langfuse's
    shape, so an OTel or Patronus trace only gets the generic fallback
    below (``output`` = last span with a non-empty output, ``steps`` =
    span count). This asymmetry is real, not an oversight: no one has
    written an equivalent metadata-reading bridge for the other two
    formats yet.

    ``overlay``, if given, is passed to ``build_overlay_fields`` along
    with the parsed spans: gold roles, fault-containment verdicts,
    the stall flag, and MAS unsafe-step counts for the four multi-agent
    metrics (``delegation_accuracy``, ``error_propagation_containment``,
    ``deadlock_livelock``, ``cumulative_unsafe_action_mas``). Unlike
    ``declared_fields``, this isn't a flat field dict: ``faults_total``
    and (when every fault has a containment verdict) ``faults_contained``
    are derived by cross-referencing the overlay's annotations against
    faults/delegations actually found in the trace, not typed directly.
    Only meaningful for a trace; raises if given alongside a payload
    that isn't one.

    ``declared_fields`` are gold/annotation fields the caller supplies
    directly (expected answer, ``should_refuse``, budgets, and so on).
    These always win over anything auto-derived from the trace, dump,
    or overlay, since the user is asserting ground truth, not a guess.

    ``audit_contamination``, only meaningful for a trace, runs
    ``audit_expected_in_inputs`` against every span's own ``input``
    text (``collect_trace_input_texts``) once ``expected`` is on the
    record. Off by default, same as the audit function's own default.
    Skipped entirely when ``declared_fields`` already asserts
    ``contaminated`` explicitly: a human's declared ground truth
    always wins over the automated check, never the other way round.
    A hit's explanation (overlap fraction, matched input text) comes
    back on ``IngestResult.contamination_evidence``, not as a new
    record field; the record itself gets ``cancelled``, which makes
    every scorer report N/A uniformly and would otherwise erase why.

    Raises
    ------
    ValueError
        ``validate_record`` rejects an unknown or mistyped field name
        (from the dump, ``overlay``, or ``declared_fields``); or
        ``overlay`` is given but ``payload`` isn't a multi-agent trace.
    """
    try:
        spans = detect_and_parse(payload)
    except ValueError:
        if overlay is not None:
            msg = "overlay requires a multi-agent trace; this payload parsed as a simple (non-trace) dump"
            raise ValueError(msg) from None
        base = simple_record_fields(payload)
        record = validate_record({**base, **(declared_fields or {})})
        return IngestResult(record=record, spans=None, trajectory=None, source_kind="simple_dump")

    base = (
        langfuse_record_fields(payload)
        if isinstance(payload.get("observations"), list)
        else _trace_fallback_fields(spans)
    )
    overlay_fields = build_overlay_fields(spans, overlay) if overlay is not None else {}
    merged = {**base, **overlay_fields, **(declared_fields or {})}
    contamination_evidence = None
    if audit_contamination and "contaminated" not in (declared_fields or {}):
        merged, contamination_evidence = audit_expected_in_inputs(
            merged, collect_trace_input_texts(spans), enabled=True
        )
    record = validate_record(merged)
    return IngestResult(
        record=record,
        spans=spans,
        trajectory=format_trajectory(spans),
        source_kind="trace",
        contamination_evidence=contamination_evidence,
    )


def ingest_inspect_eval(data: bytes, *, declared_fields: dict[str, Any] | None = None) -> list[IngestResult]:
    """Ingest an Inspect AI ``.eval`` file: one ``IngestResult`` per sample/epoch.

    Unlike ``ingest_any``, this doesn't take a parsed JSON dict: a
    ``.eval`` file is a Zstd-compressed zip archive, not plain JSON (see
    ``ingest.trace_inspect``). Each sample/epoch pair in the file becomes
    its own result: each is a genuinely separate agent run on the same
    task, not something to merge into one record.

    ``declared_fields``, when given, is applied identically to every
    resulting record (same caller-always-wins discipline as
    ``ingest_any``); there's no per-epoch override here.

    Raises
    ------
    ValueError
        The archive doesn't parse (``read_inspect_eval``), a sample has
        no ``messages`` list (``parse_inspect_sample``), or
        ``validate_record`` rejects a field name (from the sample or
        ``declared_fields``).
    """
    results = []
    for sample in read_inspect_eval(data):
        spans = parse_inspect_sample(sample)
        base = _trace_fallback_fields(spans)
        target = sample.get("target")
        if isinstance(target, str) and target.strip():
            base["expected"] = target
        record = validate_record({**base, **(declared_fields or {})})
        results.append(
            IngestResult(record=record, spans=spans, trajectory=format_trajectory(spans), source_kind="trace")
        )
    return results


def _trace_fallback_fields(spans: list[Span]) -> dict[str, Any]:
    """Return the only fields safely derivable from a bare OTel/Patronus trace.

    No gold field is ever guessed here.

    ``output`` is the last span with a non-empty output, in list order:
    the parsers already preserve pre-order (parents before children,
    earlier before later), so this is the run's final answer in the
    overwhelming majority of real traces. Never guesses ``expected``,
    safety flags, or any budget.
    """
    output: str | None = None
    for span in spans:
        if span.output:
            output = span.output
    fields: dict[str, Any] = {"steps": len(spans)}
    if output is not None:
        fields["output"] = output
    return fields


def print_ingest_any_examples() -> None:
    """Print how each of the three trace shapes and a simple dump bridge to a record."""
    otel_trace = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "t1",
                                "spanId": "s1",
                                "name": "agent_run",
                                "attributes": [{"key": "gen_ai.operation.name", "value": {"stringValue": "agent"}}],
                            },
                        ],
                    },
                ],
            },
        ],
    }
    result = ingest_any(otel_trace, declared_fields={"expected": "yes"})
    print("A bare OTel trace plus a declared expected answer:")
    print(f"  source_kind: {result.source_kind}, record: {result.record}")
    print()

    simple = {"expected": "yes", "output": "yes"}
    result = ingest_any(simple)
    print("A simple (non-trace) dump:")
    print(f"  source_kind: {result.source_kind}, record: {result.record}")


if __name__ == "__main__":
    print_ingest_any_examples()
