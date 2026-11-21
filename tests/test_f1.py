"""f1: token-overlap precision/recall F1 on [0, 1]. No new dependency."""

from aixeval.scorecard import score_f1
from aixeval.scorecard.metrics import POSTHOC_METRICS


def test_exact_match_is_full_overlap() -> None:
    result = score_f1({"expected": "the cat sat on the mat", "output": "the cat sat on the mat"})
    assert result == {
        "metric_id": "f1",
        "status": "scored",
        "raw_score": 1.0,
        "grade": 4,
    }


def test_partial_overlap_is_between_zero_and_one() -> None:
    result = score_f1({"expected": "the cat sat on the mat", "output": "a cat sat on a rug"})
    assert result["status"] == "scored"
    assert 0.0 < result["raw_score"] < 1.0


def test_unrelated_text_is_zero() -> None:
    result = score_f1({"expected": "the cat sat on the mat", "output": "quarterly revenue rose"})
    assert result["raw_score"] == 0.0
    assert result["grade"] == 0


def test_punctuation_and_case_are_normalized() -> None:
    result = score_f1({"expected": "Hello, World!", "output": "hello world"})
    assert result["raw_score"] == 1.0


def test_no_gold_is_na_not_zero() -> None:
    result = score_f1({"output": "Paris"})
    assert result == {"metric_id": "f1", "status": "na", "na_reason": "no_gold"}
    assert "raw_score" not in result


def test_missing_output_is_na() -> None:
    result = score_f1({"expected": "Paris"})
    assert result["na_reason"] == "missing_output"


def test_cancelled_beats_a_match() -> None:
    result = score_f1({"expected": "yes", "output": "yes", "cancelled": True})
    assert result["status"] == "na"
    assert result["na_reason"] == "cancelled"


def test_wrong_field_type_is_missing_evidence() -> None:
    result = score_f1({"expected": ["Paris"], "output": "Paris"})
    assert result["na_reason"] == "missing_evidence"


def test_f1_is_in_posthoc_metrics() -> None:
    assert "f1" in POSTHOC_METRICS
