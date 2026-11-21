"""meteor: stem/synonym-aware overlap on [0, 1]. Opt-in, same as bertscore.

Every test injects a fake scorer: the real nltk METEOR call needs the
WordNet corpus (a one-time download), and the unit suite must stay
hermetic regardless of whether that corpus is present.
"""

from aixeval.scorecard import score_meteor
from aixeval.scorecard.metrics import POSTHOC_METRICS


def _fake_scorer(hypothesis: str, reference: str) -> float:
    return 1.0 if hypothesis.strip().lower() == reference.strip().lower() else 0.4


def test_exact_match_scores_high() -> None:
    result = score_meteor(
        {"expected": "the cat sat on the mat", "output": "the cat sat on the mat"},
        scorer=_fake_scorer,
    )
    assert result == {
        "metric_id": "meteor",
        "status": "scored",
        "raw_score": 1.0,
        "grade": 4,
    }


def test_different_text_scores_lower() -> None:
    result = score_meteor(
        {"expected": "the cat sat on the mat", "output": "quarterly revenue rose"},
        scorer=_fake_scorer,
    )
    assert result["status"] == "scored"
    assert result["raw_score"] == 0.4


def test_scorer_output_outside_zero_one_is_clipped() -> None:
    def _overshoot_scorer(hypothesis: str, reference: str) -> float:
        return 1.4

    result = score_meteor({"expected": "yes", "output": "yes"}, scorer=_overshoot_scorer)
    assert result["raw_score"] == 1.0


def test_no_gold_is_na_not_zero() -> None:
    result = score_meteor({"output": "Paris"}, scorer=_fake_scorer)
    assert result == {"metric_id": "meteor", "status": "na", "na_reason": "no_gold"}
    assert "raw_score" not in result


def test_missing_output_is_na() -> None:
    result = score_meteor({"expected": "Paris"}, scorer=_fake_scorer)
    assert result["na_reason"] == "missing_output"


def test_cancelled_beats_a_match() -> None:
    result = score_meteor({"expected": "yes", "output": "yes", "cancelled": True}, scorer=_fake_scorer)
    assert result["status"] == "na"
    assert result["na_reason"] == "cancelled"


def test_wrong_field_type_is_missing_evidence() -> None:
    result = score_meteor({"expected": ["Paris"], "output": "Paris"}, scorer=_fake_scorer)
    assert result["na_reason"] == "missing_evidence"


def test_meteor_is_opt_in_not_in_posthoc_metrics() -> None:
    """Like bertscore, meteor never runs inside score_metrics()."""
    assert "meteor" not in POSTHOC_METRICS
