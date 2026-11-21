"""consistency is abs(run_a - run_b) / 0.30, clamped to [0, 1], or N/A."""

import pytest

from aixeval.scorecard import build_profile, score_consistency, score_metrics


def test_identical_runs_are_zero_and_grade_four() -> None:
    result = score_consistency({"run_a_score": 0.8, "run_b_score": 0.8})
    assert result == {
        "metric_id": "consistency",
        "status": "scored",
        "raw_score": 0.0,
        "grade": 4,
    }


def test_spread_is_symmetric() -> None:
    forward = score_consistency({"run_a_score": 0.9, "run_b_score": 0.75})
    backward = score_consistency({"run_a_score": 0.75, "run_b_score": 0.9})
    assert forward["raw_score"] == pytest.approx(0.5)
    assert backward["raw_score"] == pytest.approx(0.5)


def test_fifteen_point_spread_is_half_the_budget() -> None:
    result = score_consistency({"run_a_score": 0.9, "run_b_score": 0.75})
    assert result["raw_score"] == pytest.approx(0.5)
    assert result["grade"] == 2


def test_large_spread_clips_to_one() -> None:
    result = score_consistency({"run_a_score": 0.9, "run_b_score": 0.4})
    assert result["raw_score"] == 1.0
    assert result["grade"] == 0


def test_missing_or_invalid_is_na() -> None:
    assert score_consistency({"run_a_score": 0.9})["na_reason"] == "missing_evidence"
    assert score_consistency({"run_b_score": 0.9})["na_reason"] == "missing_evidence"
    assert score_consistency({"run_a_score": 1.5, "run_b_score": 0.9})["na_reason"] == "missing_evidence"


def test_bool_is_missing_evidence() -> None:
    assert score_consistency({"run_a_score": True, "run_b_score": 0.9})["na_reason"] == "missing_evidence"


def test_cancelled_beats_identical_runs() -> None:
    result = score_consistency({"run_a_score": 0.8, "run_b_score": 0.8, "cancelled": True})
    assert result["na_reason"] == "cancelled"


def test_fills_robustness_on_the_card() -> None:
    profile = build_profile(
        score_metrics(
            {
                "expected": "yes",
                "output": "yes",
                "grounding_judge": 1.0,
                "run_a_score": 0.9,
                "run_b_score": 0.4,
            }
        )
    )
    assert profile["robustness"] == 0
