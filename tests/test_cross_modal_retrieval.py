"""cross_modal_retrieval is 1.0 (rank <= k) or 0.0 (rank > k), or N/A."""

from aixeval.scorecard import build_profile, score_cross_modal_retrieval, score_metrics


def test_rank_within_k_is_one() -> None:
    result = score_cross_modal_retrieval({"retrieval_rank": 1, "retrieval_k": 5})
    assert result == {
        "metric_id": "cross_modal_retrieval",
        "status": "scored",
        "raw_score": 1.0,
        "grade": 4,
    }


def test_rank_equal_to_k_is_one() -> None:
    result = score_cross_modal_retrieval({"retrieval_rank": 5, "retrieval_k": 5})
    assert result["raw_score"] == 1.0


def test_rank_beyond_k_is_zero() -> None:
    result = score_cross_modal_retrieval({"retrieval_rank": 9, "retrieval_k": 5})
    assert result["raw_score"] == 0.0
    assert result["grade"] == 0


def test_missing_or_invalid_is_na() -> None:
    assert score_cross_modal_retrieval({"retrieval_k": 5})["na_reason"] == "missing_evidence"
    assert score_cross_modal_retrieval({"retrieval_rank": 1})["na_reason"] == "missing_evidence"
    assert score_cross_modal_retrieval({"retrieval_rank": 0, "retrieval_k": 5})["na_reason"] == "missing_evidence"
    assert score_cross_modal_retrieval({"retrieval_rank": -1, "retrieval_k": 5})["na_reason"] == "missing_evidence"


def test_bool_is_missing_evidence() -> None:
    assert score_cross_modal_retrieval({"retrieval_rank": True, "retrieval_k": 5})["na_reason"] == "missing_evidence"


def test_cancelled_beats_a_good_rank() -> None:
    result = score_cross_modal_retrieval({"retrieval_rank": 1, "retrieval_k": 5, "cancelled": True})
    assert result["na_reason"] == "cancelled"


def test_fills_capability_on_the_card() -> None:
    profile = build_profile(
        score_metrics(
            {
                "expected": "yes",
                "output": "yes",
                "grounding_judge": 1.0,
                "retrieval_rank": 9,
                "retrieval_k": 5,
            }
        )
    )
    assert profile["capability"] == 0
