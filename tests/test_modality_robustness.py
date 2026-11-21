"""modality_robustness is overage vs a 0.30 degradation budget, lower-is-better."""

import pytest

from aixeval.scorecard import build_profile, score_metrics, score_modality_robustness


def test_no_degradation_is_zero_raw_and_grade_four() -> None:
    result = score_modality_robustness({"clean_score": 0.9, "perturbed_score": 0.9})
    assert result == {
        "metric_id": "modality_robustness",
        "status": "scored",
        "raw_score": 0.0,
        "grade": 4,
    }


def test_improvement_under_perturbation_clamps_to_zero() -> None:
    result = score_modality_robustness({"clean_score": 0.8, "perturbed_score": 0.85})
    assert result["raw_score"] == 0.0
    assert result["grade"] == 4


def test_fifteen_percent_drop_is_half_the_budget() -> None:
    result = score_modality_robustness({"clean_score": 0.9, "perturbed_score": 0.75})
    assert result["raw_score"] == pytest.approx(0.5)
    assert result["grade"] == 2


def test_severe_drop_clips_to_one() -> None:
    result = score_modality_robustness({"clean_score": 0.9, "perturbed_score": 0.4})
    assert result["raw_score"] == 1.0
    assert result["grade"] == 0


def test_missing_or_invalid_is_na() -> None:
    assert score_modality_robustness({"clean_score": 0.9})["na_reason"] == "missing_evidence"
    assert score_modality_robustness({"perturbed_score": 0.9})["na_reason"] == "missing_evidence"
    assert score_modality_robustness({"clean_score": 1.5, "perturbed_score": 0.9})["na_reason"] == "missing_evidence"


def test_cancelled_beats_no_degradation() -> None:
    result = score_modality_robustness({"clean_score": 0.9, "perturbed_score": 0.9, "cancelled": True})
    assert result["na_reason"] == "cancelled"


def test_fills_robustness_on_the_card() -> None:
    profile = build_profile(
        score_metrics(
            {
                "expected": "yes",
                "output": "yes",
                "grounding_judge": 1.0,
                "clean_score": 0.9,
                "perturbed_score": 0.4,
            }
        )
    )
    assert profile["robustness"] == 0
