"""verification_correctness is 1.0 (verified_correctly) or 0.0 (not), or N/A."""

from aixeval.scorecard import build_profile, score_metrics, score_verification_correctness


def test_correct_verification_is_one() -> None:
    assert score_verification_correctness({"verified_correctly": True}) == {
        "metric_id": "verification_correctness",
        "status": "scored",
        "raw_score": 1.0,
        "grade": 4,
    }


def test_incorrect_verification_is_zero() -> None:
    result = score_verification_correctness({"verified_correctly": False})
    assert result["raw_score"] == 0.0
    assert result["grade"] == 0


def test_missing_or_non_bool_is_missing_evidence() -> None:
    assert score_verification_correctness({})["na_reason"] == "missing_evidence"
    assert score_verification_correctness({"verified_correctly": "yes"})["na_reason"] == "missing_evidence"


def test_cancelled_beats_a_correct_verification() -> None:
    assert score_verification_correctness({"verified_correctly": True, "cancelled": True})["na_reason"] == "cancelled"


def test_fills_robustness_on_the_card() -> None:
    profile = build_profile(
        score_metrics({"expected": "yes", "output": "yes", "grounding_judge": 1.0, "verified_correctly": False})
    )
    assert profile["robustness"] == 0
