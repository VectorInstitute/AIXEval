"""Normalized multi-agent trace model, tested on hand-written spans."""

import json
from pathlib import Path

import pytest

from aixeval.ingest.trace import (
    Span,
    collect_trace_input_texts,
    derive_agents,
    derive_delegations,
    derive_faults,
    derive_guardrail_counts,
    derive_messages,
    derive_stall_evidence,
    detect_and_parse,
    format_trajectory,
    validate_spans,
)


_FIXTURES = Path(__file__).parent.parent / "fixtures"


def _two_agent_run() -> list[Span]:
    return [
        Span(id="a1", kind="agent", agent="planner", input="book a flight"),
        Span(id="t1", kind="tool", parent_id="a1", input="search_flights"),
        Span(id="a2", kind="agent", parent_id="a1", agent="worker", input="find seats"),
        Span(id="l1", kind="llm", parent_id="a2", input="pick a seat"),
        Span(id="t2", kind="tool", parent_id="a2", input="seat_map"),
        Span(id="m1", kind="message", parent_id="a2", agent="worker", output="seat 14C is free"),
        Span(id="f1", kind="event", parent_id="a2", agent="worker", source_type="fault_injected"),
        Span(id="g1", kind="event", parent_id="a2", source_type="guardrail", status="ok"),
        Span(id="g2", kind="event", parent_id="a2", source_type="guardrail", status="error"),
    ]


def test_unknown_kind_or_status_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown kind"):
        Span(id="x", kind="banana")
    with pytest.raises(ValueError, match="unknown status"):
        Span(id="x", kind="tool", status="maybe")


def test_duplicate_ids_and_dangling_parents_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate span id"):
        validate_spans([Span(id="a", kind="tool"), Span(id="a", kind="tool")])
    with pytest.raises(ValueError, match="not in the run"):
        validate_spans([Span(id="a", kind="tool", parent_id="missing")])


def test_agents_and_their_step_counts() -> None:
    assert derive_agents(_two_agent_run()) == [
        {"id": "planner", "steps": 1},
        {"id": "worker", "steps": 2},
    ]


def test_collect_trace_input_texts_skips_spans_with_no_input() -> None:
    # m1/f1/g1/g2 in _two_agent_run() all have no input; only the five
    # spans that actually declare one should show up, in span order.
    assert collect_trace_input_texts(_two_agent_run()) == (
        "book a flight",
        "search_flights",
        "find seats",
        "pick a seat",
        "seat_map",
    )


def test_delegation_is_derived_from_agent_under_agent() -> None:
    assert derive_delegations(_two_agent_run()) == [
        {"sender": "planner", "recipient": "worker", "text": "find seats"},
    ]


def test_messages_combine_delegations_and_message_spans() -> None:
    messages = derive_messages(_two_agent_run())
    assert {"sender": "planner", "recipient": "worker", "text": "find seats"} in messages
    assert {"sender": "worker", "recipient": None, "text": "seat 14C is free"} in messages


def test_faults_and_guardrail_outcomes() -> None:
    spans = _two_agent_run()
    assert derive_faults(spans) == [{"span_id": "f1", "agent": "worker"}]
    assert derive_guardrail_counts(spans) == {"ok": 1, "error": 1}


def test_repeated_identical_calls_are_stall_evidence_at_threshold() -> None:
    repeated = [Span(id="a", kind="agent", agent="w")] + [
        Span(id=f"t{i}", kind="tool", parent_id="a", input="same_search") for i in range(3)
    ]
    assert derive_stall_evidence(repeated, threshold=3) == [
        {"agent": "w", "kind": "tool", "input": "same_search", "count": 3},
    ]
    assert derive_stall_evidence(repeated, threshold=4) == []


def test_stall_threshold_below_two_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        derive_stall_evidence([], threshold=1)


def test_format_trajectory_numbers_steps_and_skips_empty_spans() -> None:
    spans = [
        Span(id="a1", kind="agent", agent="worker", input="book a flight", output="booked"),
        Span(id="c1", kind="tool", source_type="TOOL", input="search", output="3 flights"),
        Span(id="c2", kind="agent", agent="worker"),
    ]
    trajectory = format_trajectory(spans)
    assert trajectory == (
        "1. [agent] worker -- input: book a flight -- output: booked\n2. [TOOL] -- input: search -- output: 3 flights"
    )


def test_format_trajectory_is_none_for_an_empty_or_silent_run() -> None:
    assert format_trajectory([]) is None
    assert format_trajectory([Span(id="a1", kind="agent", agent="worker")]) is None


def test_detect_and_parse_dispatches_to_the_right_source() -> None:
    langfuse = json.loads((_FIXTURES / "langfuse_two_agent_run.json").read_text(encoding="utf-8"))
    otel = json.loads((_FIXTURES / "otel_two_agent_run.json").read_text(encoding="utf-8"))
    patronus = json.loads((_FIXTURES / "patronus_nested_run.json").read_text(encoding="utf-8"))

    assert {s.id for s in detect_and_parse(langfuse)} == {"a1", "t1", "a2", "l1", "t2"}
    assert {s.id for s in detect_and_parse(otel)} == {
        "a1a1a1a1a1a1a1a1",
        "t1t1t1t1t1t1t1t1",
        "a2a2a2a2a2a2a2a2",
        "l1l1l1l1l1l1l1l1",
        "t2t2t2t2t2t2t2t2",
    }
    assert {s.id for s in detect_and_parse(patronus)} == {"root", "agent1", "llm1", "tool1"}


def test_detect_and_parse_rejects_an_unknown_shape() -> None:
    with pytest.raises(ValueError, match="did not match a known trace shape"):
        detect_and_parse({"something_else": []})
