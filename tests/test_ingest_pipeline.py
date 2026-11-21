"""ingest_any: detect a payload's shape, parse it, bridge to a scoreable record."""

import json
from pathlib import Path

import pytest

from aixeval.ingest.pipeline import ingest_any


FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_simple_dump_copies_known_fields() -> None:
    result = ingest_any({"expected": "yes", "output": "yes"})
    assert result.source_kind == "simple_dump"
    assert result.record == {"expected": "yes", "output": "yes"}
    assert result.spans is None
    assert result.trajectory is None


def test_simple_dump_merges_declared_fields_on_top() -> None:
    result = ingest_any({"output": "maybe"}, declared_fields={"expected": "yes", "should_refuse": False})
    assert result.record == {"output": "maybe", "expected": "yes", "should_refuse": False}


def test_langfuse_trace_uses_the_langfuse_specific_bridge() -> None:
    payload = _load("langfuse_two_agent_run.json")
    result = ingest_any(payload)
    assert result.source_kind == "trace"
    assert result.spans is not None
    assert len(result.spans) == 5
    # langfuse_record_fields: last GENERATION observation's output, steps = obs count
    assert result.record["output"] == "14C"
    assert result.record["steps"] == 5
    assert result.trajectory is not None


def test_otel_trace_uses_the_generic_fallback_with_no_fabricated_output() -> None:
    payload = _load("otel_two_agent_run.json")
    result = ingest_any(payload)
    assert result.source_kind == "trace"
    assert result.spans is not None
    assert len(result.spans) == 5
    assert result.record == {"steps": 5}  # no span carries an output in this fixture, none is guessed
    # format_trajectory skips spans with no input and no output; this fixture has
    # neither on any span, so there is genuinely nothing to show a judge.
    assert result.trajectory is None


def test_patronus_trace_uses_the_generic_fallback() -> None:
    payload = _load("patronus_nested_run.json")
    result = ingest_any(payload)
    assert result.source_kind == "trace"
    assert result.spans is not None
    assert result.record["steps"] == len(result.spans)


def test_declared_fields_override_auto_derived_trace_fields() -> None:
    payload = _load("langfuse_two_agent_run.json")
    result = ingest_any(payload, declared_fields={"output": "a human-corrected final answer"})
    assert result.record["output"] == "a human-corrected final answer"
    assert result.record["steps"] == 5  # not overridden, still auto-derived


def test_audit_contamination_off_by_default() -> None:
    payload = _load("langfuse_two_agent_run.json")
    # t1's own input is "route YYZ-LHR", a real leak, but audit_contamination
    # defaults off, matching audit_expected_in_inputs's own default.
    result = ingest_any(payload, declared_fields={"expected": "route YYZ-LHR"})
    assert "contaminated" not in result.record
    assert result.contamination_evidence is None


def test_audit_contamination_catches_a_leak_in_a_spans_own_input() -> None:
    payload = _load("langfuse_two_agent_run.json")
    result = ingest_any(payload, declared_fields={"expected": "route YYZ-LHR"}, audit_contamination=True)
    assert result.record["contaminated"] is True
    assert result.record["cancelled"] is True
    assert result.contamination_evidence is not None
    assert "route YYZ-LHR" in result.contamination_evidence


def test_audit_contamination_finds_nothing_when_expected_is_unrelated() -> None:
    payload = _load("langfuse_two_agent_run.json")
    result = ingest_any(payload, declared_fields={"expected": "completely unrelated text"}, audit_contamination=True)
    assert "contaminated" not in result.record
    assert result.contamination_evidence is None


def test_declared_contaminated_wins_over_the_audit_even_on_a_real_leak() -> None:
    payload = _load("langfuse_two_agent_run.json")
    # Same real leak as above, but the caller explicitly asserts clean --
    # a human's declared ground truth always wins, so the audit must not run.
    result = ingest_any(
        payload,
        declared_fields={"expected": "route YYZ-LHR", "contaminated": False},
        audit_contamination=True,
    )
    assert result.record["contaminated"] is False
    assert "cancelled" not in result.record
    assert result.contamination_evidence is None


def test_unknown_declared_field_raises() -> None:
    with pytest.raises(ValueError, match="nonsense_field_name"):
        ingest_any({"output": "yes"}, declared_fields={"nonsense_field_name": True})


def test_overlay_fields_are_merged_into_a_trace_record() -> None:
    payload = _load("langfuse_two_agent_run.json")
    result = ingest_any(payload, overlay={"run_stalled": True, "mas_unsafe_steps": 1, "mas_total_steps": 5})
    assert result.record["run_stalled"] is True
    assert result.record["mas_unsafe_steps"] == 1
    assert result.record["mas_total_steps"] == 5


def test_overlay_subtasks_correct_only_set_when_delegation_has_text() -> None:
    # This fixture's worker span has no input text, so build_overlay_fields
    # can't cross-reference it against a gold role; proves the overlay
    # actually derives subtasks_correct from the trace, rather than just
    # copying whatever a caller types.
    payload = _load("langfuse_two_agent_run.json")
    result = ingest_any(payload, overlay={"gold_roles": [{"subtask": "anything", "agent": "worker"}]})
    assert result.record["subtasks_total"] == 1
    assert "subtasks_correct" not in result.record


def test_declared_fields_override_overlay_fields() -> None:
    payload = _load("langfuse_two_agent_run.json")
    result = ingest_any(
        payload,
        overlay={"run_stalled": False},
        declared_fields={"run_stalled": True},
    )
    assert result.record["run_stalled"] is True


def test_overlay_on_a_simple_dump_raises() -> None:
    with pytest.raises(ValueError, match="requires a multi-agent trace"):
        ingest_any({"output": "yes"}, overlay={"run_stalled": False})


def test_invalid_overlay_raises() -> None:
    payload = _load("langfuse_two_agent_run.json")
    with pytest.raises(ValueError, match="unknown overlay"):
        ingest_any(payload, overlay={"mystery_key": 1})


def test_neither_trace_nor_simple_shape_falls_back_to_simple_dump() -> None:
    # No observations/resourceSpans/spans key at all; detect_and_parse raises,
    # so this is treated as a simple dump (consistent with ingest_dump's own
    # "copy known names, ignore the rest" rule).
    result = ingest_any({"some_other_field": "irrelevant", "output": "yes"})
    assert result.source_kind == "simple_dump"
    assert result.record == {"output": "yes"}
