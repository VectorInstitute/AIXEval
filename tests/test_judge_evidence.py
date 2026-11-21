"""Evidence extraction: join_evidence_fields (generic) + extract_evidence_langfuse.

A one-off dump shape (here: a chart-reading agent's dump, see
fixtures/chart_agent_gist.json) is not a toolkit feature: it gets a
small, local flattening helper in this test file, not a shipped
navigator. It still exercises join_evidence_fields against real,
messy nested data, same as extract_evidence_langfuse does with a real
platform.
"""

import json
from pathlib import Path
from typing import Any

from aixeval.judge.evidence import (
    extract_evidence_langfuse,
    extract_tool_calls_langfuse,
    extract_trajectory_langfuse,
    join_evidence_fields,
)


def test_join_present_fields_in_order() -> None:
    source = {"a": "first", "b": "second", "c": "third"}
    assert join_evidence_fields(source, ["c", "a"]) == "third\n\nfirst"


def test_join_skips_missing_fields() -> None:
    source = {"a": "first"}
    assert join_evidence_fields(source, ["a", "missing"]) == "first"


def test_join_returns_none_if_nothing_found() -> None:
    assert join_evidence_fields({}, ["a", "b"]) is None
    assert join_evidence_fields({"a": None}, ["a"]) is None
    assert join_evidence_fields({"a": "  "}, ["a"]) is None


def test_join_json_encodes_dict_and_list_values() -> None:
    source = {"map": [{"label": "x"}], "obj": {"k": "v"}}
    result = join_evidence_fields(source, ["map", "obj"])
    assert result == '[{"label": "x"}]\n\n{"k": "v"}'


def test_langfuse_joins_root_and_observation_inputs() -> None:
    payload = {
        "input": "root question",
        "observations": [{"input": "tool result one"}, {"input": "tool result two"}],
    }
    evidence = extract_evidence_langfuse(payload)
    assert evidence == "root question\n\ntool result one\n\ntool result two"


def test_langfuse_no_inputs_is_none() -> None:
    assert extract_evidence_langfuse({}) is None
    assert extract_evidence_langfuse({"observations": []}) is None


def test_langfuse_does_not_collect_output() -> None:
    payload = {"observations": [{"input": "in", "output": "should not appear"}]}
    evidence = extract_evidence_langfuse(payload)
    assert evidence is not None
    assert "should not appear" not in evidence


def test_trajectory_numbers_each_observation_in_order() -> None:
    payload = {
        "observations": [
            {"type": "SPAN", "name": "agent", "input": "Capital of France?", "output": "planning"},
            {"type": "GENERATION", "name": "llm", "output": "Paris"},
        ]
    }
    trajectory = extract_trajectory_langfuse(payload)
    assert trajectory == (
        "1. [SPAN] agent -- input: Capital of France? -- output: planning\n2. [GENERATION] llm -- output: Paris"
    )


def test_trajectory_shows_real_repeated_steps() -> None:
    """Same shape as the wasteful-trajectory fixture: a repeated tool call."""
    payload = {
        "observations": [
            {"type": "TOOL", "name": "search", "input": "capital of France", "output": "Paris"},
            {"type": "TOOL", "name": "search", "input": "capital of France", "output": "Paris"},
        ]
    }
    trajectory = extract_trajectory_langfuse(payload)
    assert trajectory is not None
    assert trajectory.count("[TOOL] search") == 2


def test_trajectory_no_observations_is_none() -> None:
    assert extract_trajectory_langfuse({}) is None
    assert extract_trajectory_langfuse({"observations": []}) is None


def test_trajectory_skips_non_dict_observations() -> None:
    payload = {"observations": ["not a dict", {"type": "SPAN"}]}
    trajectory = extract_trajectory_langfuse(payload)
    assert trajectory == "2. [SPAN]"


def test_trajectory_against_the_real_wasteful_fixture() -> None:
    path = Path(__file__).parent.parent / "fixtures" / "langfuse_wasteful_trajectory.json"
    payload = json.loads(path.read_text())
    trajectory = extract_trajectory_langfuse(payload)
    assert trajectory is not None
    assert trajectory.count("[TOOL] search") == 3
    assert trajectory.startswith("1. [SPAN] agent")
    assert trajectory.splitlines()[-1].startswith("5. [GENERATION] llm")


def test_tool_calls_filters_to_tool_type_only() -> None:
    payload = {
        "observations": [
            {"type": "SPAN", "name": "agent", "input": "weather in Paris in F", "output": "planning"},
            {"type": "TOOL", "name": "get_weather", "input": {"city": "Paris"}, "output": {"temp_c": 20}},
            {"type": "GENERATION", "name": "llm", "output": "It's about 68F in Paris."},
        ]
    }
    assert extract_tool_calls_langfuse(payload) == [{"tool": "get_weather", "args": {"city": "Paris"}}]


def test_tool_calls_missing_input_defaults_to_empty_dict() -> None:
    payload = {"observations": [{"type": "TOOL", "name": "ping"}]}
    assert extract_tool_calls_langfuse(payload) == [{"tool": "ping", "args": {}}]


def test_tool_calls_non_dict_input_defaults_to_empty_dict() -> None:
    payload = {"observations": [{"type": "TOOL", "name": "ping", "input": "not a dict"}]}
    assert extract_tool_calls_langfuse(payload) == [{"tool": "ping", "args": {}}]


def test_tool_calls_skips_entries_with_no_usable_name() -> None:
    payload = {"observations": [{"type": "TOOL", "input": {}}, {"type": "TOOL", "name": "", "input": {}}]}
    assert extract_tool_calls_langfuse(payload) is None


def test_tool_calls_preserves_order() -> None:
    payload = {
        "observations": [
            {"type": "TOOL", "name": "first", "input": {}},
            {"type": "SPAN", "name": "agent"},
            {"type": "TOOL", "name": "second", "input": {}},
        ]
    }
    assert extract_tool_calls_langfuse(payload) == [
        {"tool": "first", "args": {}},
        {"tool": "second", "args": {}},
    ]


def test_tool_calls_no_observations_is_none() -> None:
    assert extract_tool_calls_langfuse({}) is None
    assert extract_tool_calls_langfuse({"observations": []}) is None
    assert extract_tool_calls_langfuse({"observations": [{"type": "SPAN"}]}) is None


# --- A worked example against a real, messy, one-off dump shape ---
# Not a toolkit feature: this flattening lives here, not in judge/evidence.py.


def _flatten_chart_agent_dump(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten one project's chart-reading agent dump. Test-only, not shipped."""
    as_dict = lambda v: v if isinstance(v, dict) else {}  # noqa: E731
    ocr = as_dict(payload.get("ocr"))
    legend = as_dict(payload.get("legend_grounding"))
    metadata = as_dict(as_dict(payload.get("sample")).get("metadata"))
    return {
        "ocr_text": ocr.get("parsed"),
        "legend_map": legend.get("legend_map"),
        "analyst_caption": metadata.get("verified_caption"),
        "related_sentences": metadata.get("related_sentences"),
    }


_CHART_AGENT_FIELDS = ("ocr_text", "legend_map", "analyst_caption", "related_sentences")


def test_chart_agent_shape_excludes_vision_and_verifier() -> None:
    payload = {
        "sample": {"metadata": {"verified_caption": "Fig. 10"}},
        "vision": {"parsed": {"answer": "the model's own guess, not evidence"}},
        "verifier": {"parsed": {"reasoning": "the model's own reasoning, not evidence"}},
    }
    evidence = join_evidence_fields(_flatten_chart_agent_dump(payload), _CHART_AGENT_FIELDS)
    assert evidence == "Fig. 10"
    assert "own guess" not in evidence
    assert "own reasoning" not in evidence


def test_chart_agent_shape_against_a_tiny_fixture() -> None:
    path = Path(__file__).parent.parent / "fixtures" / "chart_agent_gist.json"
    payload = json.loads(path.read_text())
    evidence = join_evidence_fields(_flatten_chart_agent_dump(payload), _CHART_AGENT_FIELDS)
    assert evidence is not None
    assert '"chart_type": "bar"' in evidence
    assert "Fig. 10: Mexico Sales" in evidence
    assert "own guess" not in evidence
    assert "own reasoning" not in evidence
