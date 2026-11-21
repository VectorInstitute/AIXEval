"""Patronus/TRAIL export -> normalized spans, tested on a synthetic committed fixture.

The real nested shape (one top span, the run under child_spans,
openinference.span.kind inside a flat span_attributes dict) was verified
against a real TRAIL export.
"""

import json
from pathlib import Path

import pytest

from aixeval.ingest.trace import derive_agents, format_trajectory
from aixeval.ingest.trace_patronus import parse_patronus_trace


_FIXTURE = Path(__file__).parent.parent / "fixtures" / "patronus_nested_run.json"


def _payload() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def test_child_spans_are_flattened_with_parent_ids_preserved() -> None:
    spans = {s.id: s for s in parse_patronus_trace(_payload())}
    assert set(spans) == {"root", "agent1", "llm1", "tool1"}
    assert spans["agent1"].parent_id == "root"
    assert spans["llm1"].parent_id == "agent1"
    assert spans["tool1"].parent_id == "agent1"


def test_kind_comes_from_the_openinference_attribute_not_top_level_span_kind() -> None:
    spans = {s.id: s for s in parse_patronus_trace(_payload())}
    assert spans["agent1"].kind == "agent"
    assert spans["llm1"].kind == "llm"
    assert spans["tool1"].kind == "tool"
    assert spans["root"].kind == "other"


def test_agent_identity_is_left_none_not_guessed() -> None:
    spans = {s.id: s for s in parse_patronus_trace(_payload())}
    assert spans["agent1"].agent is None
    assert derive_agents(parse_patronus_trace(_payload())) == []


def test_status_code_maps_to_ok_and_error() -> None:
    spans = {s.id: s for s in parse_patronus_trace(_payload())}
    assert spans["agent1"].status == "ok"
    assert spans["tool1"].status == "error"


def test_input_and_output_come_from_span_attributes() -> None:
    spans = {s.id: s for s in parse_patronus_trace(_payload())}
    assert spans["tool1"].input == "route YYZ-LHR"
    assert spans["tool1"].output == "timeout"
    assert spans["root"].input is None


def test_source_type_is_the_openinference_kind_or_the_span_name() -> None:
    spans = {s.id: s for s in parse_patronus_trace(_payload())}
    assert spans["llm1"].source_type == "LLM"
    assert spans["root"].source_type == "main"


def test_format_trajectory_skips_spans_with_no_input_or_output() -> None:
    trajectory = format_trajectory(parse_patronus_trace(_payload()))
    assert trajectory is not None
    assert "main" not in trajectory
    assert "route YYZ-LHR" in trajectory
    assert "call search_flights" in trajectory


def test_missing_spans_list_is_rejected() -> None:
    with pytest.raises(ValueError, match="no spans list"):
        parse_patronus_trace({})


def test_span_without_span_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="no span_id"):
        parse_patronus_trace({"spans": [{"span_name": "main"}]})
