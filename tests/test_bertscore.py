"""bertscore: DeBERTa-backed contextual F1 on [0, 1]. Opt-in, not in score_metrics().

Every test injects a fake scorer: the real ``bert_score.score`` needs
torch/transformers and downloads ``microsoft/deberta-xlarge-mnli`` on first
use, and the unit suite must stay fast and hermetic.
"""

from aixeval.scorecard import score_bertscore
from aixeval.scorecard.metrics import POSTHOC_METRICS


def _fake_scorer(cands: list[str], refs: list[str]) -> tuple[list[float], list[float], list[float]]:
    same = [1.0 if c.strip().lower() == r.strip().lower() else 0.3 for c, r in zip(cands, refs, strict=True)]
    return same, same, same


def test_exact_match_scores_full_f1() -> None:
    result = score_bertscore(
        {"expected": "the cat sat on the mat", "output": "the cat sat on the mat"},
        scorer=_fake_scorer,
    )
    assert result == {
        "metric_id": "bertscore",
        "status": "scored",
        "raw_score": 1.0,
        "grade": 4,
    }


def test_different_text_scores_partial_f1() -> None:
    result = score_bertscore(
        {"expected": "the cat sat on the mat", "output": "quarterly revenue rose"},
        scorer=_fake_scorer,
    )
    assert result["status"] == "scored"
    assert result["raw_score"] == 0.3


def test_scorer_output_outside_zero_one_is_clipped() -> None:
    """The injected scorer can return anything; we clip before grading."""

    def _overshoot_scorer(cands: list[str], refs: list[str]) -> tuple[list[float], list[float], list[float]]:
        return [1.0], [1.0], [1.2]

    result = score_bertscore({"expected": "yes", "output": "yes"}, scorer=_overshoot_scorer)
    assert result["raw_score"] == 1.0


def test_no_gold_is_na_not_zero() -> None:
    result = score_bertscore({"output": "Paris"}, scorer=_fake_scorer)
    assert result == {"metric_id": "bertscore", "status": "na", "na_reason": "no_gold"}
    assert "raw_score" not in result


def test_missing_output_is_na() -> None:
    result = score_bertscore({"expected": "Paris"}, scorer=_fake_scorer)
    assert result["na_reason"] == "missing_output"


def test_cancelled_beats_a_match() -> None:
    result = score_bertscore({"expected": "yes", "output": "yes", "cancelled": True}, scorer=_fake_scorer)
    assert result["status"] == "na"
    assert result["na_reason"] == "cancelled"


def test_wrong_field_type_is_missing_evidence() -> None:
    result = score_bertscore({"expected": ["Paris"], "output": "Paris"}, scorer=_fake_scorer)
    assert result["na_reason"] == "missing_evidence"


def test_bertscore_is_opt_in_not_in_posthoc_metrics() -> None:
    """Unlike bleu/rouge/wer, bertscore never runs inside score_metrics()."""
    assert "bertscore" not in POSTHOC_METRICS
