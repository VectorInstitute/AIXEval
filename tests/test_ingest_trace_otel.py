"""OTLP JSON -> normalized spans, tested on a synthetic committed fixture.

The same two-agent run is also in fixtures/langfuse_two_agent_run.json.
Both parsers should give the same derivations for it.
"""

import json
from pathlib import Path

import pytest

from aixeval.ingest.trace import (
    derive_agents,
    derive_faults,
    derive_messages,
)
from aixeval.ingest.trace_langfuse import parse_langfuse_trace
from aixeval.ingest.trace_otel import parse_otel_trace


_FIXTURES = Path(__file__).parent.parent / "fixtures"


def _otel_payload() -> dict:
    return json.loads((_FIXTURES / "otel_two_agent_run.json").read_text(encoding="utf-8"))


def _langfuse_payload() -> dict:
    return json.loads((_FIXTURES / "langfuse_two_agent_run.json").read_text(encoding="utf-8"))


def test_kinds_and_parents_follow_the_genai_operation_name() -> None:
    spans = {s.id: s for s in parse_otel_trace(_otel_payload())}
    assert spans["a1a1a1a1a1a1a1a1"].kind == "agent"
    assert spans["t1t1t1t1t1t1t1t1"].kind == "tool"
    assert spans["l1l1l1l1l1l1l1l1"].kind == "llm"
    assert spans["a2a2a2a2a2a2a2a2"].parent_id == "a1a1a1a1a1a1a1a1"
    assert spans["a1a1a1a1a1a1a1a1"].parent_id is None


def test_agent_identity_is_the_gen_ai_agent_name_attribute() -> None:
    spans = {s.id: s for s in parse_otel_trace(_otel_payload())}
    assert spans["a1a1a1a1a1a1a1a1"].agent == "planner"
    assert spans["a2a2a2a2a2a2a2a2"].agent == "worker"
    assert spans["t1t1t1t1t1t1t1t1"].agent is None


def test_status_codes_map_to_ok_and_error() -> None:
    spans = {s.id: s for s in parse_otel_trace(_otel_payload())}
    assert spans["a1a1a1a1a1a1a1a1"].status == "ok"
    assert spans["a2a2a2a2a2a2a2a2"].status == "error"


def test_delegation_text_is_left_empty_not_guessed() -> None:
    spans = parse_otel_trace(_otel_payload())
    assert all(s.input is None and s.output is None for s in spans if s.kind == "agent")


def test_span_events_become_event_spans_with_the_event_name_as_source_type() -> None:
    payload = _otel_payload()
    worker = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][2]
    worker["events"] = [{"name": "fault_injected", "timeUnixNano": "1788280805500000000"}]
    spans = parse_otel_trace(payload)
    assert derive_faults(spans) == [{"span_id": "a2a2a2a2a2a2a2a2:event:0", "agent": None}]


def test_otel_and_langfuse_give_the_same_derivations_for_the_same_run() -> None:
    otel = parse_otel_trace(_otel_payload())
    langfuse = parse_langfuse_trace(_langfuse_payload())
    assert derive_agents(otel) == derive_agents(langfuse)
    assert derive_messages(otel) == derive_messages(langfuse)


def test_two_trace_ids_in_one_payload_are_rejected() -> None:
    payload = _otel_payload()
    payload["resourceSpans"][0]["scopeSpans"][0]["spans"][1]["traceId"] = "ffffffffffffffffffffffffffffffff"
    with pytest.raises(ValueError, match="one run is one trace"):
        parse_otel_trace(payload)


def test_missing_resource_spans_is_rejected() -> None:
    with pytest.raises(ValueError, match="no resourceSpans"):
        parse_otel_trace({})
