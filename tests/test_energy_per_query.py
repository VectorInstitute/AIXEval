"""energy_per_query is overage vs a kWh budget, lower-is-better, or N/A."""

import pytest

from aixeval.scorecard import build_profile, score_energy_per_query, score_metrics


def test_under_budget_is_zero_raw_and_grade_four() -> None:
    result = score_energy_per_query({"energy_kwh": 0.0003, "energy_budget_kwh": 0.001})
    assert result == {
        "metric_id": "energy_per_query",
        "status": "scored",
        "raw_score": 0.0,
        "grade": 4,
    }


def test_on_budget_is_zero_raw() -> None:
    result = score_energy_per_query({"energy_kwh": 0.001, "energy_budget_kwh": 0.001})
    assert result["raw_score"] == 0.0
    assert result["grade"] == 4


def test_twenty_five_percent_over_is_point_two_five() -> None:
    result = score_energy_per_query({"energy_kwh": 0.00025, "energy_budget_kwh": 0.0002})
    assert result["raw_score"] == pytest.approx(0.25)
    assert result["grade"] == 3


def test_double_or_more_clips_to_one() -> None:
    result = score_energy_per_query({"energy_kwh": 0.003, "energy_budget_kwh": 0.001})
    assert result["raw_score"] == 1.0
    assert result["grade"] == 0


def test_missing_or_invalid_is_na() -> None:
    assert score_energy_per_query({"energy_kwh": 0.0003})["na_reason"] == "missing_evidence"
    assert score_energy_per_query({"energy_budget_kwh": 0.001})["na_reason"] == "missing_evidence"
    assert (
        score_energy_per_query({"energy_kwh": -0.0001, "energy_budget_kwh": 0.001})["na_reason"] == "missing_evidence"
    )
    assert score_energy_per_query({"energy_kwh": 0.0003, "energy_budget_kwh": 0.0})["na_reason"] == "missing_evidence"


def test_cancelled_beats_a_low_energy_call() -> None:
    result = score_energy_per_query({"energy_kwh": 0.0001, "energy_budget_kwh": 0.001, "cancelled": True})
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
                "energy_kwh": 0.00025,
                "energy_budget_kwh": 0.0002,
            }
        )
    )
    assert profile["efficiency"] == 3
    assert profile["applied_hard_stops"] == ()
