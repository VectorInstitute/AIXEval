"""bleu: sacrebleu n-gram overlap on [0, 1], then a higher-is-better grade."""

from aixeval.scorecard import score_bleu


def test_exact_match_scores_high() -> None:
    result = score_bleu({"expected": "the cat sat on the mat", "output": "the cat sat on the mat"})
    assert result["status"] == "scored"
    assert result["raw_score"] > 0.99
    assert result["grade"] == 4


def test_unrelated_text_scores_low() -> None:
    result = score_bleu({"expected": "the cat sat on the mat", "output": "quarterly revenue rose"})
    assert result["status"] == "scored"
    assert result["raw_score"] < 0.20
    assert result["grade"] == 0


def test_raw_score_stays_in_unit_range() -> None:
    result = score_bleu({"expected": "the cat sat on the mat", "output": "the cat is sitting on the mat"})
    assert 0.0 <= result["raw_score"] <= 1.0


def test_no_gold_is_na_not_zero() -> None:
    result = score_bleu({"output": "Paris"})
    assert result == {"metric_id": "bleu", "status": "na", "na_reason": "no_gold"}
    assert "raw_score" not in result


def test_missing_output_is_na() -> None:
    result = score_bleu({"expected": "Paris"})
    assert result["na_reason"] == "missing_output"


def test_cancelled_beats_a_match() -> None:
    result = score_bleu({"expected": "yes", "output": "yes", "cancelled": True})
    assert result["status"] == "na"
    assert result["na_reason"] == "cancelled"


def test_wrong_field_type_is_missing_evidence() -> None:
    result = score_bleu({"expected": ["Paris"], "output": "Paris"})
    assert result["na_reason"] == "missing_evidence"
