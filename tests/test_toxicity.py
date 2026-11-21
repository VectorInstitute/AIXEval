"""toxicity: classifier-backed probability on [0, 1], lower-is-better. Opt-in.

Every test injects a fake classifier: the real ``unitary/toxic-bert``
call needs ``transformers`` and downloads a model on first use, and
the unit suite must stay fast and hermetic.
"""

from aixeval.scorecard import score_toxicity
from aixeval.scorecard.metrics import POSTHOC_METRICS


def _fake_classifier(text: str) -> float:
    return 0.9 if "toxic" in text.lower() else 0.05


def test_non_toxic_output_scores_low_raw_and_grade_four() -> None:
    result = score_toxicity({"output": "Thank you for your help!"}, classifier=_fake_classifier)
    assert result == {
        "metric_id": "toxicity",
        "status": "scored",
        "raw_score": 0.05,
        "grade": 4,
    }


def test_toxic_output_scores_high_raw_and_grade_zero() -> None:
    result = score_toxicity({"output": "this is a toxic reply"}, classifier=_fake_classifier)
    assert result["status"] == "scored"
    assert result["raw_score"] == 0.9
    assert result["grade"] == 0


def test_classifier_output_outside_zero_one_is_clipped() -> None:
    """The injected classifier can return anything; we clip before grading."""

    def _overshoot_classifier(text: str) -> float:
        return 1.4

    result = score_toxicity({"output": "hello"}, classifier=_overshoot_classifier)
    assert result["raw_score"] == 1.0


def test_no_output_is_na_not_zero() -> None:
    result = score_toxicity({"expected": "yes"}, classifier=_fake_classifier)
    assert result == {"metric_id": "toxicity", "status": "na", "na_reason": "missing_output"}
    assert "raw_score" not in result


def test_blank_output_is_na() -> None:
    result = score_toxicity({"output": "   "}, classifier=_fake_classifier)
    assert result["na_reason"] == "missing_output"


def test_wrong_field_type_is_missing_evidence() -> None:
    result = score_toxicity({"output": ["not text"]}, classifier=_fake_classifier)
    assert result["na_reason"] == "missing_evidence"


def test_cancelled_beats_a_clean_output() -> None:
    result = score_toxicity({"output": "hello", "cancelled": True}, classifier=_fake_classifier)
    assert result["status"] == "na"
    assert result["na_reason"] == "cancelled"


def test_no_gold_needed_reference_free() -> None:
    """Unlike bertscore, toxicity never reads expected: reference-free."""
    result = score_toxicity({"output": "Thank you for your help!"}, classifier=_fake_classifier)
    assert result["status"] == "scored"


def test_toxicity_is_opt_in_not_in_posthoc_metrics() -> None:
    """Unlike bleu/rouge/wer, toxicity never runs inside score_metrics()."""
    assert "toxicity" not in POSTHOC_METRICS


def test_stored_verdict_wins_over_classifier() -> None:
    """A verdict already on the record, from either mechanism, is graded directly."""
    result = score_toxicity({"toxicity_judge": 0.1, "output": "this is a toxic reply"}, classifier=_fake_classifier)
    assert result["raw_score"] == 0.1
    assert result["grade"] == 4


def test_stored_verdict_needs_no_output_or_classifier() -> None:
    result = score_toxicity({"toxicity_judge": 0.3})
    assert result["status"] == "scored"
    assert result["raw_score"] == 0.3


def test_stored_verdict_out_of_range_is_clipped() -> None:
    """Same convention as the classifier path: clip, not reject."""
    assert score_toxicity({"toxicity_judge": 1.5})["raw_score"] == 1.0
    assert score_toxicity({"toxicity_judge": -0.1})["raw_score"] == 0.0


def test_stored_verdict_bool_is_na() -> None:
    assert score_toxicity({"toxicity_judge": True})["na_reason"] == "missing_evidence"


def test_cancelled_beats_a_stored_verdict() -> None:
    result = score_toxicity({"toxicity_judge": 0.1, "cancelled": True})
    assert result["na_reason"] == "cancelled"
