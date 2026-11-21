"""verbalized_confidence grades a stated [0, 1], or N/A if none."""

from aixeval.scorecard import build_profile, score_metrics, score_verbalized_confidence


def test_sure_is_one() -> None:
    result = score_verbalized_confidence({"stated_confidence": 1.0})
    assert result == {
        "metric_id": "verbalized_confidence",
        "status": "scored",
        "raw_score": 1.0,
        "grade": 4,
    }


def test_partial_keeps_the_raw_score() -> None:
    result = score_verbalized_confidence({"stated_confidence": 0.7})
    assert result["raw_score"] == 0.7
    assert result["grade"] == 3


def test_zero_stated_is_scored_not_na() -> None:
    result = score_verbalized_confidence({"stated_confidence": 0.0})
    assert result["status"] == "scored"
    assert result["raw_score"] == 0.0
    assert result["grade"] == 0


def test_missing_statement_is_na_not_zero() -> None:
    result = score_verbalized_confidence({"output": "yes"})
    assert result == {
        "metric_id": "verbalized_confidence",
        "status": "na",
        "na_reason": "missing_evidence",
    }
    assert "raw_score" not in result


def test_invalid_statement_is_na() -> None:
    assert score_verbalized_confidence({"stated_confidence": True})["na_reason"] == "missing_evidence"
    assert score_verbalized_confidence({"stated_confidence": 1.2})["na_reason"] == "missing_evidence"


def test_cancelled_beats_a_statement() -> None:
    result = score_verbalized_confidence({"stated_confidence": 1.0, "cancelled": True})
    assert result["na_reason"] == "cancelled"


def test_transparency_comes_from_stated_confidence() -> None:
    profile = build_profile(
        score_metrics(
            {
                "expected": "yes",
                "output": "yes",
                "grounding_judge": 1.0,
                "had_failure": False,
                "contaminated": False,
                "high_impact_in_scope": False,
                "stated_confidence": 1.0,
            }
        )
    )
    assert profile["transparency"] == 4
    assert profile["fairness"] is None
