"""step_efficiency is overage vs a declared path length, or N/A."""

from aixeval.scorecard import build_profile, score_metrics, score_step_efficiency


def test_under_budget_is_zero_raw_and_grade_four() -> None:
    result = score_step_efficiency({"steps": 8, "step_budget": 10})
    assert result == {
        "metric_id": "step_efficiency",
        "status": "scored",
        "raw_score": 0.0,
        "grade": 4,
    }


def test_on_budget_is_zero_raw() -> None:
    result = score_step_efficiency({"steps": 10, "step_budget": 10})
    assert result["raw_score"] == 0.0
    assert result["grade"] == 4


def test_twenty_percent_over_is_point_two() -> None:
    result = score_step_efficiency({"steps": 12, "step_budget": 10})
    assert result["raw_score"] == 0.2
    assert result["grade"] == 3


def test_double_or_more_clips_to_one() -> None:
    result = score_step_efficiency({"steps": 21, "step_budget": 8})
    assert result["raw_score"] == 1.0
    assert result["grade"] == 0


def test_missing_or_invalid_is_na() -> None:
    assert score_step_efficiency({"steps": 8})["na_reason"] == "missing_evidence"
    assert score_step_efficiency({"step_budget": 10})["na_reason"] == "missing_evidence"
    assert score_step_efficiency({"steps": 8.0, "step_budget": 10})["na_reason"] == "missing_evidence"
    assert score_step_efficiency({"steps": -1, "step_budget": 10})["na_reason"] == "missing_evidence"
    assert score_step_efficiency({"steps": 8, "step_budget": 0})["na_reason"] == "missing_evidence"


def test_cancelled_beats_a_short_path() -> None:
    result = score_step_efficiency({"steps": 3, "step_budget": 10, "cancelled": True})
    assert result["na_reason"] == "cancelled"


def test_efficiency_is_min_of_latency_and_steps() -> None:
    profile = build_profile(
        score_metrics(
            {
                "expected": "yes",
                "output": "yes",
                "grounding_judge": 1.0,
                "had_failure": False,
                "contaminated": False,
                "high_impact_in_scope": False,
                "latency_sec": 31.0,
                "latency_budget_sec": 60.0,
                "steps": 21,
                "step_budget": 8,
            }
        )
    )
    assert profile["efficiency"] == 0
    assert profile["applied_hard_stops"] == ()
