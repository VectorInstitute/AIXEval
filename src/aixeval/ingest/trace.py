"""Normalized multi-agent trace model and the derivations metrics read from it.

Source parsers (Langfuse, OpenTelemetry, Patronus/TRAIL) map into ``Span``.
Everything below works on that shape alone, so no metric depends on a
source format. Derivations are pure functions of the span list. Ground
truth that no trace carries (gold roles, fault containment, baselines)
is not here.

Adding a new source means writing one function, ``payload -> list[Span]``,
and registering it in ``detect_and_parse``. Each source's own kind
vocabulary and shape differ enough (a flat list, OTLP's triple nesting,
Patronus's recursive tree) that a single declarative mapping can't absorb
all three, so each parser keeps its own walk, sharing only ``text_or_none``
and the ``Span``/derivation contract below.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


KINDS: frozenset[str] = frozenset({"agent", "tool", "llm", "delegation", "message", "event", "other"})
STATUSES: frozenset[str] = frozenset({"ok", "error"})
FAULT_EVENT_TYPE = "fault_injected"
GUARDRAIL_EVENT_TYPE = "guardrail"


def text_or_none(value: object) -> str | None:
    """Keep string values only. Shared by every parser; non-strings are dropped."""
    return value if isinstance(value, str) else None


@dataclass(frozen=True)
class Span:
    """One normalized span. ``kind`` and ``status`` are checked on construction."""

    id: str
    kind: str
    parent_id: str | None = None
    agent: str | None = None
    status: str | None = None
    source_type: str | None = None
    input: str | None = None
    output: str | None = None
    start: str | None = None
    end: str | None = None

    def __post_init__(self) -> None:
        """Reject an unknown kind or status before it reaches a derivation."""
        if self.kind not in KINDS:
            msg = f"span {self.id!r}: unknown kind {self.kind!r}"
            raise ValueError(msg)
        if self.status is not None and self.status not in STATUSES:
            msg = f"span {self.id!r}: unknown status {self.status!r}"
            raise ValueError(msg)


def validate_spans(spans: list[Span]) -> None:
    """Check ids are unique and every parent_id points at a span in the run."""
    ids = [span.id for span in spans]
    duplicates = sorted({sid for sid, count in Counter(ids).items() if count > 1})
    if duplicates:
        msg = f"duplicate span id(s): {', '.join(duplicates)}"
        raise ValueError(msg)
    known = set(ids)
    for span in spans:
        if span.parent_id is not None and span.parent_id not in known:
            msg = f"span {span.id!r}: parent {span.parent_id!r} is not in the run"
            raise ValueError(msg)


def _by_id(spans: list[Span]) -> dict[str, Span]:
    return {span.id: span for span in spans}


def _nearest_agent_ancestor(span: Span, index: dict[str, Span]) -> str | None:
    """Return the agent owning ``span``: its own, else the nearest ancestor's."""
    current: Span | None = span
    while current is not None:
        if current.kind == "agent" and current.agent:
            return current.agent
        current = index.get(current.parent_id) if current.parent_id else None
    return None


def derive_agents(spans: list[Span]) -> list[dict[str, Any]]:
    """One entry per agent, with the tool and llm spans it owned as its step count."""
    validate_spans(spans)
    index = _by_id(spans)
    names = sorted({span.agent for span in spans if span.kind == "agent" and span.agent})
    steps: Counter[str] = Counter()
    for span in spans:
        if span.kind in ("tool", "llm"):
            owner = _nearest_agent_ancestor(span, index)
            if owner is not None:
                steps[owner] += 1
    return [{"id": name, "steps": steps[name]} for name in names]


def collect_trace_input_texts(spans: list[Span]) -> tuple[str, ...]:
    """Every span's own ``input`` text: what ``audit_expected_in_inputs`` searches.

    Mirrors ``ingest.langfuse.collect_langfuse_input_texts`` for the
    newer multi-agent ``Span`` model: a single flat list of input
    strings, not re-walking raw payload shapes per source format.
    """
    validate_spans(spans)
    return tuple(span.input for span in spans if span.input)


def derive_delegations(spans: list[Span]) -> list[dict[str, Any]]:
    """Agent spans whose parent is a different agent's span: one handoff each."""
    validate_spans(spans)
    index = _by_id(spans)
    handoffs = []
    for span in spans:
        if span.kind != "agent" or not span.agent or span.parent_id is None:
            continue
        parent = index[span.parent_id]
        sender = _nearest_agent_ancestor(parent, index)
        if sender is not None and sender != span.agent:
            handoffs.append({"sender": sender, "recipient": span.agent, "text": span.input})
    return handoffs


def derive_messages(spans: list[Span]) -> list[dict[str, Any]]:
    """Return delegations plus explicit message spans.

    A message span has no derivable recipient (open question 5), so it is None.
    """
    messages = [
        {"sender": d["sender"], "recipient": d["recipient"], "text": d["text"]} for d in derive_delegations(spans)
    ]
    for span in spans:
        if span.kind == "message":
            messages.append({"sender": span.agent, "recipient": None, "text": span.output})
    return messages


def derive_faults(spans: list[Span]) -> list[dict[str, Any]]:
    """Injected-fault events. Containment is not in the trace; the overlay sets it."""
    return [
        {"span_id": span.id, "agent": span.agent}
        for span in spans
        if span.kind == "event" and span.source_type == FAULT_EVENT_TYPE
    ]


def derive_guardrail_counts(spans: list[Span]) -> dict[str, int]:
    """Guardrail check events, split by outcome: ``ok`` is pass, ``error`` is fail."""
    counts: Counter[str] = Counter()
    for span in spans:
        if span.kind == "event" and span.source_type == GUARDRAIL_EVENT_TYPE:
            counts[span.status or "unknown"] += 1
    return dict(counts)


def derive_stall_evidence(spans: list[Span], threshold: int) -> list[dict[str, Any]]:
    """Identical repeated tool or llm calls under one agent, at or above ``threshold``.

    The threshold is declared. This returns evidence only; the declared
    ``run_stalled`` flag is what a metric grades.
    """
    if threshold < 2:
        msg = "stall threshold must be at least 2"
        raise ValueError(msg)
    validate_spans(spans)
    index = _by_id(spans)
    groups: Counter[tuple[str, str, str]] = Counter()
    for span in spans:
        if span.kind in ("tool", "llm"):
            owner = _nearest_agent_ancestor(span, index)
            if owner is not None:
                groups[(owner, span.kind, span.input or "")] += 1
    return [
        {"agent": agent, "kind": kind, "input": text, "count": count}
        for (agent, kind, text), count in sorted(groups.items())
        if count >= threshold
    ]


def format_trajectory(spans: list[Span], *, max_field_chars: int = 2000) -> str | None:
    """Return an ordered, readable trace of the steps taken, for a judge to read.

    Source-agnostic: works on any parser's output, since ``trajectory_optimality``'s
    judge only needs plain text (same shape as ``extract_trajectory_langfuse``).
    Spans list order is used as step order. A span with no input and no
    output is skipped, the same discipline ``_format_step`` already uses
    for Langfuse's own equivalent.

    max_field_chars
        Cap each span's ``input``/``output`` text at this many characters.
        Real instrumentation (e.g. an LLM-call span's logged ``input``)
        routinely embeds the full system prompt plus the entire prior
        conversation so far on every step; uncapped, that makes the
        joined trajectory text grow roughly with the square of step
        count. On a real multi-step agent trace that reaches tens of
        thousands of tokens, which is slow to judge and can blow past a
        judge model's effective context. Each field is truncated to its
        leading ``max_field_chars`` characters with a marker noting how
        much was cut; short fields (the common case in tests and small
        traces) are unaffected.
    """
    validate_spans(spans)
    lines: list[str] = []
    for index, span in enumerate(spans, start=1):
        if span.input is None and span.output is None:
            continue
        label = f"[{span.source_type}]" if span.source_type else f"[{span.kind}]"
        who = f" {span.agent}" if span.agent else ""
        parts = [f"{index}. {label}{who}"]
        if span.input is not None:
            parts.append(f"input: {_truncate_field(span.input, max_field_chars)}")
        if span.output is not None:
            parts.append(f"output: {_truncate_field(span.output, max_field_chars)}")
        lines.append(" -- ".join(parts))
    return "\n".join(lines) if lines else None


def _truncate_field(text: str, limit: int) -> str:
    """Keep the leading ``limit`` chars of ``text``, with a cut marker if trimmed."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]} ... [truncated, {len(text)} chars total]"


def detect_and_parse(payload: dict[str, Any]) -> list[Span]:
    """Pick the right parser for ``payload``'s shape, then parse.

    Checked in this order: Langfuse (an ``observations`` list), OTLP
    (a ``resourceSpans`` list), tau-bench (a ``traj`` list), then
    Patronus/TRAIL (a ``spans`` list), Patronus last, since its
    top-level key is the most generic-looking of the four. Raises
    ``ValueError`` if none match.
    """
    from aixeval.ingest.trace_langfuse import parse_langfuse_trace  # noqa: PLC0415
    from aixeval.ingest.trace_otel import parse_otel_trace  # noqa: PLC0415
    from aixeval.ingest.trace_patronus import parse_patronus_trace  # noqa: PLC0415
    from aixeval.ingest.trace_taubench import parse_taubench_trace  # noqa: PLC0415

    if isinstance(payload.get("observations"), list):
        return parse_langfuse_trace(payload)
    if isinstance(payload.get("resourceSpans"), list):
        return parse_otel_trace(payload)
    if isinstance(payload.get("traj"), list):
        return parse_taubench_trace(payload)
    if isinstance(payload.get("spans"), list):
        return parse_patronus_trace(payload)
    msg = "payload did not match a known trace shape (Langfuse/OTel/Patronus/tau-bench)"
    raise ValueError(msg)
