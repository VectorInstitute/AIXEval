"""sub_goal_progress is subgoals_completed / subgoals_total, clipped to [0, 1]."""

from aixeval.scorecard import build_profile, score_metrics, score_sub_goal_progress


def test_all_subgoals_completed_is_one() -> None:
    result = score_sub_goal_progress({"subgoals_completed": 4, "subgoals_total": 4})
    assert result == {
        "metric_id": "sub_goal_progress",
        "status": "scored",
        "raw_score": 1.0,
        "grade": 4,
    }


def test_partial_completion_is_the_fraction() -> None:
    result = score_sub_goal_progress({"subgoals_completed": 1, "subgoals_total": 4})
    assert result["raw_score"] == 0.25
    assert result["grade"] == 1


def test_none_completed_is_zero() -> None:
    result = score_sub_goal_progress({"subgoals_completed": 0, "subgoals_total": 4})
    assert result["raw_score"] == 0.0
    assert result["grade"] == 0


def test_completed_above_total_is_clamped_to_one() -> None:
    result = score_sub_goal_progress({"subgoals_completed": 6, "subgoals_total": 4})
    assert result["raw_score"] == 1.0


def test_missing_or_invalid_is_na() -> None:
    assert score_sub_goal_progress({"subgoals_total": 4})["na_reason"] == "missing_evidence"
    assert score_sub_goal_progress({"subgoals_completed": 2})["na_reason"] == "missing_evidence"
    assert score_sub_goal_progress({"subgoals_completed": 2, "subgoals_total": 0})["na_reason"] == "missing_evidence"
    assert score_sub_goal_progress({"subgoals_completed": -1, "subgoals_total": 4})["na_reason"] == "missing_evidence"


def test_bool_is_missing_evidence() -> None:
    result = score_sub_goal_progress({"subgoals_completed": True, "subgoals_total": 4})
    assert result["na_reason"] == "missing_evidence"


def test_cancelled_beats_full_completion() -> None:
    result = score_sub_goal_progress({"subgoals_completed": 4, "subgoals_total": 4, "cancelled": True})
    assert result["na_reason"] == "cancelled"


def test_fills_capability_on_the_card() -> None:
    profile = build_profile(
        score_metrics(
            {
                "expected": "yes",
                "output": "yes",
                "grounding_judge": 1.0,
                "subgoals_completed": 0,
                "subgoals_total": 4,
            }
        )
    )
    assert profile["capability"] == 0
