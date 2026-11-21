"""contamination is 1.0 (leak) or 0.0 (clean), lower-is-better, or N/A."""

from aixeval.scorecard import build_profile, score_contamination, score_metrics


def test_clean_is_zero_raw_and_grade_four() -> None:
    result = score_contamination({"contaminated": False})
    assert result == {
        "metric_id": "contamination",
        "status": "scored",
        "raw_score": 0.0,
        "grade": 4,
    }


def test_leak_is_one_raw_and_grade_zero() -> None:
    result = score_contamination({"contaminated": True})
    assert result["status"] == "scored"
    assert result["raw_score"] == 1.0
    assert result["grade"] == 0


def test_missing_flag_is_na_not_zero() -> None:
    result = score_contamination({"output": "yes"})
    assert result == {
        "metric_id": "contamination",
        "status": "na",
        "na_reason": "missing_evidence",
    }
    assert "raw_score" not in result
    assert "grade" not in result


def test_non_bool_is_missing_evidence() -> None:
    assert score_contamination({"contaminated": 1})["na_reason"] == "missing_evidence"


def test_cancelled_beats_a_clean_flag() -> None:
    result = score_contamination({"contaminated": False, "cancelled": True})
    assert result["status"] == "na"
    assert result["na_reason"] == "cancelled"


def test_leak_voids_capability_on_the_card() -> None:
    profile = build_profile(
        score_metrics(
            {
                "expected": "yes",
                "output": "yes",
                "grounding_judge": 1.0,
                "had_failure": False,
                "contaminated": True,
            }
        )
    )
    assert profile["capability"] is None
    assert profile["governance"] == 0
    assert profile["applied_hard_stops"] == ("contamination_void",)
