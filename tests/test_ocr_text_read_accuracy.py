"""ocr_text_read_accuracy is 1 - CER, higher-is-better, or N/A."""

from aixeval.scorecard import build_profile, score_metrics, score_ocr_text_read_accuracy


def test_exact_match_is_one() -> None:
    result = score_ocr_text_read_accuracy({"expected": "INVOICE #4471", "output": "INVOICE #4471"})
    assert result == {
        "metric_id": "ocr_text_read_accuracy",
        "status": "scored",
        "raw_score": 1.0,
        "grade": 4,
    }


def test_one_character_misread_is_partial() -> None:
    result = score_ocr_text_read_accuracy({"expected": "INVOICE #4471", "output": "INVOICE #447I"})
    assert result["status"] == "scored"
    assert 0.0 < result["raw_score"] < 1.0


def test_unrelated_text_clips_to_zero() -> None:
    result = score_ocr_text_read_accuracy({"expected": "A", "output": "a completely different string"})
    assert result["raw_score"] == 0.0
    assert result["grade"] == 0


def test_no_gold_is_na_not_zero() -> None:
    result = score_ocr_text_read_accuracy({"output": "INVOICE #4471"})
    assert result == {"metric_id": "ocr_text_read_accuracy", "status": "na", "na_reason": "no_gold"}
    assert "raw_score" not in result


def test_missing_output_is_na() -> None:
    result = score_ocr_text_read_accuracy({"expected": "INVOICE #4471"})
    assert result["na_reason"] == "missing_output"


def test_cancelled_beats_a_match() -> None:
    result = score_ocr_text_read_accuracy({"expected": "INVOICE #4471", "output": "INVOICE #4471", "cancelled": True})
    assert result["status"] == "na"
    assert result["na_reason"] == "cancelled"


def test_wrong_field_type_is_missing_evidence() -> None:
    result = score_ocr_text_read_accuracy({"expected": ["not text"], "output": "INVOICE #4471"})
    assert result["na_reason"] == "missing_evidence"


def test_fills_capability_on_the_card() -> None:
    profile = build_profile(
        score_metrics(
            {
                "expected": "INVOICE #4471",
                "output": "a completely different string",
            }
        )
    )
    assert profile["capability"] == 0
