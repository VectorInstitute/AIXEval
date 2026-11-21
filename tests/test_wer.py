"""wer: word error rate clipped to [0, 1], then a lower-is-better grade."""

from aixeval.scorecard import score_wer


def test_exact_match_is_zero_error() -> None:
    result = score_wer({"expected": "the cat sat on the mat", "output": "the cat sat on the mat"})
    assert result == {
        "metric_id": "wer",
        "status": "scored",
        "raw_score": 0.0,
        "grade": 4,
    }


def test_one_word_dropped_is_partial_error() -> None:
    result = score_wer({"expected": "the cat sat on the mat", "output": "the cat sat on mat"})
    assert result["status"] == "scored"
    assert 0.0 < result["raw_score"] < 1.0


def test_wildly_different_text_is_clipped_to_one() -> None:
    """jiwer.wer is unbounded above 1.0 on a much longer hypothesis; we clip."""
    result = score_wer({"expected": "a", "output": "a completely different sentence here now"})
    assert result["raw_score"] == 1.0
    assert result["grade"] == 0


def test_no_gold_is_na_not_zero() -> None:
    result = score_wer({"output": "Paris"})
    assert result == {"metric_id": "wer", "status": "na", "na_reason": "no_gold"}
    assert "raw_score" not in result


def test_missing_output_is_na() -> None:
    result = score_wer({"expected": "Paris"})
    assert result["na_reason"] == "missing_output"


def test_cancelled_beats_a_match() -> None:
    result = score_wer({"expected": "yes", "output": "yes", "cancelled": True})
    assert result["status"] == "na"
    assert result["na_reason"] == "cancelled"


def test_wrong_field_type_is_missing_evidence() -> None:
    result = score_wer({"expected": ["Paris"], "output": "Paris"})
    assert result["na_reason"] == "missing_evidence"
