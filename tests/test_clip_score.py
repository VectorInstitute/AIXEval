"""clip_score: CLIP embedding similarity, on [0, 1]. Opt-in, not in score_metrics().

Every test injects a fake scorer: the real CLIP call needs
transformers/Pillow and downloads a model on first use, and the unit
suite must stay fast and hermetic.
"""

from aixeval.scorecard import score_clip_score
from aixeval.scorecard.metrics import POSTHOC_METRICS


def _fake_scorer(text: str, image_path: str) -> float:
    return 0.9 if "dog" in text.lower() and "dog" in image_path.lower() else 0.2


def test_matching_text_and_image_scores_high() -> None:
    result = score_clip_score({"output": "a dog on a beach", "image_path": "/data/dog_beach.jpg"}, scorer=_fake_scorer)
    assert result == {
        "metric_id": "clip_score",
        "status": "scored",
        "raw_score": 0.9,
        "grade": 4,
    }


def test_mismatched_text_and_image_scores_low() -> None:
    result = score_clip_score({"output": "a dog on a beach", "image_path": "/data/cat_couch.jpg"}, scorer=_fake_scorer)
    assert result["status"] == "scored"
    assert result["raw_score"] == 0.2


def test_scorer_output_outside_zero_one_is_clipped() -> None:
    """The real CLIPScore can exceed 1; any injected scorer's output is clipped."""

    def _overshoot_scorer(text: str, image_path: str) -> float:
        return 1.8

    result = score_clip_score({"output": "x", "image_path": "/data/x.jpg"}, scorer=_overshoot_scorer)
    assert result["raw_score"] == 1.0


def test_no_image_path_is_na_not_zero() -> None:
    result = score_clip_score({"output": "a dog on a beach"}, scorer=_fake_scorer)
    assert result == {"metric_id": "clip_score", "status": "na", "na_reason": "no_gold"}
    assert "raw_score" not in result


def test_missing_output_is_na() -> None:
    result = score_clip_score({"image_path": "/data/dog_beach.jpg"}, scorer=_fake_scorer)
    assert result["na_reason"] == "missing_output"


def test_cancelled_beats_a_match() -> None:
    result = score_clip_score(
        {"output": "a dog on a beach", "image_path": "/data/dog_beach.jpg", "cancelled": True}, scorer=_fake_scorer
    )
    assert result["status"] == "na"
    assert result["na_reason"] == "cancelled"


def test_wrong_field_type_is_missing_evidence() -> None:
    result = score_clip_score({"output": ["not text"], "image_path": "/data/dog_beach.jpg"}, scorer=_fake_scorer)
    assert result["na_reason"] == "missing_evidence"


def test_clip_score_is_opt_in_not_in_posthoc_metrics() -> None:
    """Unlike bleu/rouge/wer, clip_score never runs inside score_metrics()."""
    assert "clip_score" not in POSTHOC_METRICS
