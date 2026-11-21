"""fairness_gap is 1 - min(rates) / max(rates) over declared group_outcomes, or N/A."""

import pytest

from aixeval.scorecard import build_profile, score_fairness_gap, score_metrics


def test_equal_rates_are_zero_and_grade_four() -> None:
    result = score_fairness_gap({"group_outcomes": {"a": 0.5, "b": 0.5}})
    assert result == {
        "metric_id": "fairness_gap",
        "status": "scored",
        "raw_score": 0.0,
        "grade": 4,
    }


def test_just_below_the_four_fifths_threshold_scores_good() -> None:
    result = score_fairness_gap({"group_outcomes": {"a": 0.5, "b": 0.39}})
    assert result["raw_score"] == pytest.approx(0.22)
    assert result["grade"] == 3


def test_just_above_the_four_fifths_threshold_scores_excellent() -> None:
    result = score_fairness_gap({"group_outcomes": {"a": 0.5, "b": 0.41}})
    assert result["raw_score"] == pytest.approx(0.18)
    assert result["grade"] == 4


def test_clear_adverse_impact_scores_worse() -> None:
    result = score_fairness_gap({"group_outcomes": {"a": 0.6, "b": 0.3}})
    assert result["raw_score"] == pytest.approx(0.5)
    assert result["grade"] == 2


def test_three_groups_uses_the_widest_pair() -> None:
    result = score_fairness_gap({"group_outcomes": {"a": 0.9, "b": 0.8, "c": 0.4}})
    assert result["raw_score"] == pytest.approx(1.0 - 0.4 / 0.9)


def test_all_zero_rates_is_no_gap_not_a_penalty() -> None:
    result = score_fairness_gap({"group_outcomes": {"a": 0.0, "b": 0.0}})
    assert result["raw_score"] == 0.0
    assert result["grade"] == 4


def test_missing_or_invalid_is_na() -> None:
    assert score_fairness_gap({})["na_reason"] == "missing_evidence"
    assert score_fairness_gap({"group_outcomes": {"a": 0.5}})["na_reason"] == "missing_evidence"
    assert score_fairness_gap({"group_outcomes": {"a": 1.5, "b": 0.5}})["na_reason"] == "missing_evidence"
    assert score_fairness_gap({"group_outcomes": [0.5, 0.4]})["na_reason"] == "missing_evidence"


def test_bool_rate_is_missing_evidence() -> None:
    assert score_fairness_gap({"group_outcomes": {"a": True, "b": 0.5}})["na_reason"] == "missing_evidence"


def test_cancelled_beats_equal_rates() -> None:
    result = score_fairness_gap({"group_outcomes": {"a": 0.5, "b": 0.5}, "cancelled": True})
    assert result["na_reason"] == "cancelled"


def test_fills_fairness_on_the_card() -> None:
    profile = build_profile(
        score_metrics(
            {
                "expected": "yes",
                "output": "yes",
                "grounding_judge": 1.0,
                "group_outcomes": {"a": 0.6, "b": 0.3},
            }
        )
    )
    assert profile["fairness"] == 2


def test_without_group_outcomes_fairness_stays_na() -> None:
    profile = build_profile(score_metrics({"expected": "yes", "output": "yes"}))
    assert profile["fairness"] is None
