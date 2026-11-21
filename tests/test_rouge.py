"""rouge: ROUGE-L F-measure on [0, 1], then a higher-is-better grade."""

from aixeval.scorecard import score_rouge


def test_exact_match_scores_high() -> None:
    result = score_rouge({"expected": "the cat sat on the mat", "output": "the cat sat on the mat"})
    assert result["status"] == "scored"
    assert result["raw_score"] == 1.0
    assert result["grade"] == 4


def test_unrelated_text_scores_low() -> None:
    result = score_rouge({"expected": "the cat sat on the mat", "output": "quarterly revenue rose sharply"})
    assert result["status"] == "scored"
    assert result["raw_score"] < 0.20
    assert result["grade"] == 0


def test_raw_score_stays_in_unit_range() -> None:
    result = score_rouge({"expected": "the cat sat on the mat", "output": "the cat is sitting on the mat"})
    assert 0.0 <= result["raw_score"] <= 1.0


def test_no_gold_is_na_not_zero() -> None:
    result = score_rouge({"output": "Paris"})
    assert result == {"metric_id": "rouge", "status": "na", "na_reason": "no_gold"}
    assert "raw_score" not in result


def test_missing_output_is_na() -> None:
    result = score_rouge({"expected": "Paris"})
    assert result["na_reason"] == "missing_output"


def test_cancelled_beats_a_match() -> None:
    result = score_rouge({"expected": "yes", "output": "yes", "cancelled": True})
    assert result["status"] == "na"
    assert result["na_reason"] == "cancelled"


def test_wrong_field_type_is_missing_evidence() -> None:
    result = score_rouge({"expected": ["Paris"], "output": "Paris"})
    assert result["na_reason"] == "missing_evidence"
