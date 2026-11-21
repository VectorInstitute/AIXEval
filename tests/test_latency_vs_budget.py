"""latency_vs_budget is overage vs budget, lower-is-better, or N/A."""

from aixeval.scorecard import build_profile, score_latency_vs_budget, score_metrics


def test_under_budget_is_zero_raw_and_grade_four() -> None:
    result = score_latency_vs_budget({"latency_sec": 31.0, "latency_budget_sec": 60.0})
    assert result == {
        "metric_id": "latency_vs_budget",
        "status": "scored",
        "raw_score": 0.0,
        "grade": 4,
    }


def test_on_budget_is_zero_raw() -> None:
    result = score_latency_vs_budget({"latency_sec": 60, "latency_budget_sec": 60})
    assert result["raw_score"] == 0.0
    assert result["grade"] == 4


def test_twenty_percent_over_is_point_two() -> None:
    result = score_latency_vs_budget({"latency_sec": 72.0, "latency_budget_sec": 60.0})
    assert result["raw_score"] == 0.2
    assert result["grade"] == 3


def test_double_or_more_clips_to_one() -> None:
    result = score_latency_vs_budget({"latency_sec": 176.0, "latency_budget_sec": 60.0})
    assert result["raw_score"] == 1.0
    assert result["grade"] == 0


def test_missing_or_invalid_is_na() -> None:
    assert score_latency_vs_budget({"latency_sec": 31.0})["na_reason"] == "missing_evidence"
    assert score_latency_vs_budget({"latency_budget_sec": 60.0})["na_reason"] == "missing_evidence"
    assert score_latency_vs_budget({"latency_sec": -1.0, "latency_budget_sec": 60.0})["na_reason"] == (
        "missing_evidence"
    )
    assert score_latency_vs_budget({"latency_sec": 31.0, "latency_budget_sec": 0.0})["na_reason"] == (
        "missing_evidence"
    )


def test_cancelled_beats_a_fast_run() -> None:
    result = score_latency_vs_budget({"latency_sec": 1.0, "latency_budget_sec": 60.0, "cancelled": True})
    assert result["na_reason"] == "cancelled"


def test_overage_fills_efficiency_on_the_card() -> None:
    profile = build_profile(
        score_metrics(
            {
                "expected": "yes",
                "output": "yes",
                "grounding_judge": 1.0,
                "had_failure": False,
                "contaminated": False,
                "high_impact_in_scope": False,
                "latency_sec": 72.0,
                "latency_budget_sec": 60.0,
            }
        )
    )
    assert profile["efficiency"] == 3
    assert profile["applied_hard_stops"] == ()
