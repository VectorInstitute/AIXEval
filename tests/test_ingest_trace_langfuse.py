"""Langfuse export -> normalized spans, tested on a synthetic committed fixture."""

import json
from pathlib import Path

import pytest

from aixeval.ingest.trace import (
    derive_agents,
    derive_faults,
    derive_guardrail_counts,
    derive_messages,
    derive_stall_evidence,
)
from aixeval.ingest.trace_langfuse import parse_langfuse_trace


_FIXTURE = Path(__file__).parent.parent / "fixtures" / "langfuse_two_agent_run.json"


def _payload() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def test_kinds_and_parents_come_from_observation_type_and_parent_id() -> None:
    spans = {s.id: s for s in parse_langfuse_trace(_payload())}
    assert spans["a1"].kind == "agent"
    assert spans["t1"].kind == "tool"
    assert spans["l1"].kind == "llm"
    assert spans["a2"].parent_id == "a1"
    assert spans["a1"].parent_id is None


def test_agent_identity_is_the_metadata_attribute_not_the_name_wrapper() -> None:
    spans = {s.id: s for s in parse_langfuse_trace(_payload())}
    assert spans["a1"].agent == "planner"
    assert spans["a2"].agent == "worker"
    assert spans["t1"].agent is None


def test_name_template_is_the_fallback_when_metadata_is_missing() -> None:
    payload = _payload()
    payload["observations"][0]["metadata"] = {}
    spans = {s.id: s for s in parse_langfuse_trace(payload)}
    assert spans["a1"].agent == "planner"


def test_level_maps_to_status() -> None:
    spans = {s.id: s for s in parse_langfuse_trace(_payload())}
    assert spans["a1"].status == "ok"
    assert spans["a2"].status == "error"
    assert spans["t2"].status == "error"


def test_non_string_input_is_dropped_not_stringified() -> None:
    spans = {s.id: s for s in parse_langfuse_trace(_payload())}
    assert spans["a1"].input is None
    assert spans["t1"].input == "route YYZ-LHR"


def test_source_type_keeps_the_langfuse_type_string() -> None:
    spans = {s.id: s for s in parse_langfuse_trace(_payload())}
    assert spans["l1"].source_type == "GENERATION"


def test_event_observations_use_their_name_as_source_type_not_the_bare_type() -> None:
    # Langfuse's own type values ("EVENT"/"GUARDRAIL") only ever say "this is
    # an event", same OTel convention (span-event "name" field), confirmed
    # working via test_span_events_become_event_spans_with_the_event_name_as_source_type
    # in test_ingest_trace_otel.py. Before this, Langfuse-shaped traces had no
    # way to produce a fault_injected/guardrail span at all: kind and
    # source_type were both derived from the same bare `type` field, so a
    # literal type="fault_injected" would map to kind="other" (unrecognized),
    # and type="EVENT" would give source_type="EVENT" (not the specific kind).
    payload = _payload()
    payload["observations"].append(
        {
            "id": "ev1",
            "type": "EVENT",
            "name": "fault_injected",
            "parentObservationId": "a2",
            "level": "DEFAULT",
            "statusMessage": None,
            "input": None,
            "output": "a fault was injected here",
            "startTime": "2026-10-01T10:00:06Z",
            "endTime": "2026-10-01T10:00:06Z",
            "metadata": {},
        }
    )
    spans = {s.id: s for s in parse_langfuse_trace(payload)}
    assert spans["ev1"].kind == "event"
    assert spans["ev1"].source_type == "fault_injected"
    assert derive_faults(parse_langfuse_trace(payload)) == [{"span_id": "ev1", "agent": None}]


def test_derivations_run_on_parsed_spans() -> None:
    spans = parse_langfuse_trace(_payload())
    assert derive_agents(spans) == [
        {"id": "planner", "steps": 1},
        {"id": "worker", "steps": 2},
    ]
    assert {"sender": "planner", "recipient": "worker", "text": None} in derive_messages(spans)
    assert derive_faults(spans) == []
    assert derive_guardrail_counts(spans) == {}
    assert derive_stall_evidence(spans, threshold=2) == []


def test_missing_observations_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="no observations list"):
        parse_langfuse_trace({"id": "x"})


def test_observation_without_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="no id"):
        parse_langfuse_trace({"observations": [{"type": "TOOL"}]})


def test_dangling_parent_is_rejected() -> None:
    payload = _payload()
    payload["observations"][1]["parentObservationId"] = "ghost"
    with pytest.raises(ValueError, match="not in the run"):
        parse_langfuse_trace(payload)
