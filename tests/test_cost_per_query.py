"""cost_per_query is overage vs a dollar budget, lower-is-better, or N/A."""

from aixeval.scorecard import build_profile, score_cost_per_query, score_metrics


def test_under_budget_is_zero_raw_and_grade_four() -> None:
    result = score_cost_per_query({"cost_usd": 0.003, "cost_budget_usd": 0.01})
    assert result == {
        "metric_id": "cost_per_query",
        "status": "scored",
        "raw_score": 0.0,
        "grade": 4,
    }


def test_on_budget_is_zero_raw() -> None:
    result = score_cost_per_query({"cost_usd": 0.01, "cost_budget_usd": 0.01})
    assert result["raw_score"] == 0.0
    assert result["grade"] == 4


def test_twenty_percent_over_is_point_two() -> None:
    result = score_cost_per_query({"cost_usd": 0.012, "cost_budget_usd": 0.01})
    assert result["raw_score"] == 0.2
    assert result["grade"] == 3


def test_double_or_more_clips_to_one() -> None:
    result = score_cost_per_query({"cost_usd": 0.03, "cost_budget_usd": 0.01})
    assert result["raw_score"] == 1.0
    assert result["grade"] == 0


def test_missing_or_invalid_is_na() -> None:
    assert score_cost_per_query({"cost_usd": 0.003})["na_reason"] == "missing_evidence"
    assert score_cost_per_query({"cost_budget_usd": 0.01})["na_reason"] == "missing_evidence"
    assert score_cost_per_query({"cost_usd": -0.001, "cost_budget_usd": 0.01})["na_reason"] == "missing_evidence"
    assert score_cost_per_query({"cost_usd": 0.003, "cost_budget_usd": 0.0})["na_reason"] == "missing_evidence"


def test_cancelled_beats_a_cheap_call() -> None:
    result = score_cost_per_query({"cost_usd": 0.001, "cost_budget_usd": 0.01, "cancelled": True})
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
                "cost_usd": 0.012,
                "cost_budget_usd": 0.01,
            }
        )
    )
    assert profile["efficiency"] == 3
    assert profile["applied_hard_stops"] == ()
