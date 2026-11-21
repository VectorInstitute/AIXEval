"""Overlay: trace plus annotation give flat fields, and the four MAS metrics."""

import pytest

from aixeval.ingest.overlay import build_overlay_fields, validate_overlay
from aixeval.ingest.trace import Span
from aixeval.scorecard import (
    score_cumulative_unsafe_action_mas,
    score_deadlock_livelock,
    score_delegation_accuracy,
    score_error_propagation_containment,
)


def _run_with_fault_and_handoff(handoff_text: str | None = "find seats") -> list[Span]:
    return [
        Span(id="a1", kind="agent", agent="planner", input="book"),
        Span(id="a2", kind="agent", parent_id="a1", agent="worker", input=handoff_text),
        Span(id="f1", kind="event", parent_id="a2", agent="worker", source_type="fault_injected"),
    ]


def test_validate_rejects_unknown_keys_and_bad_types() -> None:
    with pytest.raises(ValueError, match="unknown overlay"):
        validate_overlay({"mystery": 1})
    with pytest.raises(ValueError, match="contained"):
        validate_overlay({"faults": [{"span_id": "f1", "contained": "yes"}]})
    with pytest.raises(ValueError, match="run_stalled"):
        validate_overlay({"run_stalled": 1})


def test_containment_is_built_only_when_every_fault_has_a_verdict() -> None:
    spans = _run_with_fault_and_handoff()
    fields = build_overlay_fields(spans, {"faults": [{"span_id": "f1", "contained": True}]})
    assert fields["faults_total"] == 1
    assert fields["faults_contained"] == 1
    assert "faults_contained" not in build_overlay_fields(spans, {})


def test_delegation_accuracy_needs_handoff_text() -> None:
    gold = {"gold_roles": [{"subtask": "find seats", "agent": "worker"}]}
    with_text = build_overlay_fields(_run_with_fault_and_handoff("find seats"), gold)
    assert with_text["subtasks_correct"] == 1
    without_text = build_overlay_fields(_run_with_fault_and_handoff(None), gold)
    assert "subtasks_correct" not in without_text
    assert score_delegation_accuracy(without_text)["na_reason"] == "missing_evidence"


def test_delegation_matches_on_text_and_recipient() -> None:
    gold = {"gold_roles": [{"subtask": "find seats", "agent": "planner"}]}
    fields = build_overlay_fields(_run_with_fault_and_handoff("find seats"), gold)
    assert fields["subtasks_correct"] == 0


def test_containment_metric_na_when_no_faults_injected() -> None:
    spans = [Span(id="a1", kind="agent", agent="planner")]
    fields = build_overlay_fields(spans, {})
    assert score_error_propagation_containment(fields)["na_reason"] == "not_in_scope"


def test_containment_metric_scores_the_share() -> None:
    result = score_error_propagation_containment({"faults_total": 4, "faults_contained": 3})
    assert result["raw_score"] == 0.75
    assert result["grade"] == 3


def test_deadlock_metric_is_lower_is_better() -> None:
    assert score_deadlock_livelock({"run_stalled": True})["grade"] == 0
    assert score_deadlock_livelock({"run_stalled": False})["grade"] == 4
    assert score_deadlock_livelock({})["na_reason"] == "missing_evidence"


def test_unsafe_share_is_lower_is_better_and_clamped() -> None:
    result = score_cumulative_unsafe_action_mas({"mas_unsafe_steps": 3, "mas_total_steps": 10})
    assert result["raw_score"] == pytest.approx(0.3)
    assert score_cumulative_unsafe_action_mas({"mas_unsafe_steps": 1, "mas_total_steps": 0})["na_reason"] == (
        "missing_evidence"
    )


def test_cancelled_beats_a_result_in_every_mas_metric() -> None:
    assert score_deadlock_livelock({"run_stalled": True, "cancelled": True})["na_reason"] == "cancelled"
    assert (
        score_delegation_accuracy({"subtasks_correct": 1, "subtasks_total": 1, "cancelled": True})["na_reason"]
        == "cancelled"
    )
