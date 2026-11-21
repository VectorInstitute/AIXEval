"""caption_quality is CIDEr (TF-IDF n-gram cosine similarity), higher-is-better."""

import pytest

from aixeval.scorecard import build_profile, score_caption_quality, score_metrics


def test_exact_match_to_single_reference_is_one() -> None:
    """Regression test: a single reference used to zero out IDF entirely."""
    result = score_caption_quality(
        {"output": "a dog running on the beach", "reference_captions": ["a dog running on the beach"]}
    )
    assert result == {
        "metric_id": "caption_quality",
        "status": "scored",
        "raw_score": 1.0,
        "grade": 4,
    }


def test_close_paraphrase_scores_lower_than_exact_match() -> None:
    result = score_caption_quality(
        {
            "output": "a dog runs along the beach",
            "reference_captions": ["a dog running on the beach", "a dog runs on the sand"],
        }
    )
    assert result["status"] == "scored"
    assert 0.0 < result["raw_score"] < 1.0


def test_unrelated_caption_scores_lower_than_paraphrase() -> None:
    paraphrase = score_caption_quality(
        {
            "output": "a dog runs along the beach",
            "reference_captions": ["a dog running on the beach", "a dog runs on the sand"],
        }
    )
    unrelated = score_caption_quality(
        {"output": "a cat sleeping on a couch", "reference_captions": ["a dog running on the beach"]}
    )
    assert unrelated["raw_score"] < paraphrase["raw_score"]


def test_identical_candidate_and_single_reference_with_short_text() -> None:
    """A short exact match (fewer tokens than the longest n-gram) still scores 1.0."""
    result = score_caption_quality({"output": "a dog", "reference_captions": ["a dog"]})
    assert result["raw_score"] == pytest.approx(1.0)


def test_no_reference_captions_is_na_not_zero() -> None:
    result = score_caption_quality({"output": "a dog running on the beach"})
    assert result == {"metric_id": "caption_quality", "status": "na", "na_reason": "no_gold"}
    assert "raw_score" not in result


def test_missing_output_is_na() -> None:
    result = score_caption_quality({"reference_captions": ["a dog running on the beach"]})
    assert result["na_reason"] == "missing_output"


def test_empty_reference_list_is_missing_evidence() -> None:
    result = score_caption_quality({"output": "a dog running on the beach", "reference_captions": []})
    assert result["na_reason"] == "missing_evidence"


def test_cancelled_beats_a_match() -> None:
    result = score_caption_quality(
        {
            "output": "a dog running on the beach",
            "reference_captions": ["a dog running on the beach"],
            "cancelled": True,
        }
    )
    assert result["na_reason"] == "cancelled"


def test_fills_capability_on_the_card() -> None:
    profile = build_profile(
        score_metrics(
            {
                "expected": "yes",
                "output": "a cat sleeping on a couch",
                "grounding_judge": 1.0,
                "reference_captions": ["a dog running on the beach"],
            }
        )
    )
    assert profile["capability"] == 0
