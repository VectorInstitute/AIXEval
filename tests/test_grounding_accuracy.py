"""grounding_accuracy is 1.0 (IoU >= 0.5) or 0.0 (IoU < 0.5), or N/A."""

from aixeval.scorecard import build_profile, score_grounding_accuracy, score_metrics


def test_exact_box_match_is_one() -> None:
    result = score_grounding_accuracy({"predicted_bbox": [10, 10, 50, 50], "gold_bbox": [10, 10, 50, 50]})
    assert result == {
        "metric_id": "grounding_accuracy",
        "status": "scored",
        "raw_score": 1.0,
        "grade": 4,
    }


def test_mostly_overlapping_box_is_one() -> None:
    result = score_grounding_accuracy({"predicted_bbox": [12, 12, 52, 52], "gold_bbox": [10, 10, 50, 50]})
    assert result["raw_score"] == 1.0


def test_barely_overlapping_box_is_zero() -> None:
    result = score_grounding_accuracy({"predicted_bbox": [45, 45, 85, 85], "gold_bbox": [10, 10, 50, 50]})
    assert result["raw_score"] == 0.0
    assert result["grade"] == 0


def test_non_overlapping_box_is_zero() -> None:
    result = score_grounding_accuracy({"predicted_bbox": [100, 100, 150, 150], "gold_bbox": [10, 10, 50, 50]})
    assert result["raw_score"] == 0.0


def test_no_gold_box_is_na_not_zero() -> None:
    result = score_grounding_accuracy({"predicted_bbox": [10, 10, 50, 50]})
    assert result == {"metric_id": "grounding_accuracy", "status": "na", "na_reason": "no_gold"}
    assert "raw_score" not in result


def test_missing_predicted_box_is_missing_output() -> None:
    result = score_grounding_accuracy({"gold_bbox": [10, 10, 50, 50]})
    assert result["na_reason"] == "missing_output"


def test_malformed_box_is_missing_evidence() -> None:
    assert (
        score_grounding_accuracy({"predicted_bbox": [1, 2, 3], "gold_bbox": [10, 10, 50, 50]})["na_reason"]
        == "missing_evidence"
    )
    assert (
        score_grounding_accuracy({"predicted_bbox": "not a box", "gold_bbox": [10, 10, 50, 50]})["na_reason"]
        == "missing_evidence"
    )


def test_cancelled_beats_a_match() -> None:
    result = score_grounding_accuracy(
        {"predicted_bbox": [10, 10, 50, 50], "gold_bbox": [10, 10, 50, 50], "cancelled": True}
    )
    assert result["na_reason"] == "cancelled"


def test_fills_capability_on_the_card() -> None:
    profile = build_profile(
        score_metrics(
            {
                "expected": "yes",
                "output": "yes",
                "grounding_judge": 1.0,
                "predicted_bbox": [100, 100, 150, 150],
                "gold_bbox": [10, 10, 50, 50],
            }
        )
    )
    assert profile["capability"] == 0
