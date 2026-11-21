"""vqa_accuracy is VQAv2 soft accuracy (min(1, matches/3)), higher-is-better, or N/A."""

from aixeval.scorecard import build_profile, score_metrics, score_vqa_accuracy


def test_all_annotators_agree_is_one() -> None:
    result = score_vqa_accuracy({"output": "yellow", "gold_answers": ["yellow"] * 10})
    assert result == {
        "metric_id": "vqa_accuracy",
        "status": "scored",
        "raw_score": 1.0,
        "grade": 4,
    }


def test_exactly_three_of_ten_is_fully_correct() -> None:
    result = score_vqa_accuracy({"output": "yellow", "gold_answers": ["yellow"] * 3 + ["gold"] * 7})
    assert result["raw_score"] == 1.0
    assert result["grade"] == 4


def test_one_of_ten_is_partial_credit() -> None:
    result = score_vqa_accuracy({"output": "yellow", "gold_answers": ["yellow"] + ["gold"] * 9})
    assert result["raw_score"] == 1 / 3
    assert result["grade"] == 1


def test_no_annotator_match_is_zero() -> None:
    result = score_vqa_accuracy({"output": "blue", "gold_answers": ["yellow"] * 10})
    assert result["raw_score"] == 0.0
    assert result["grade"] == 0


def test_normalization_ignores_case_and_punctuation() -> None:
    result = score_vqa_accuracy({"output": "Yellow!", "gold_answers": ["yellow"] * 3})
    assert result["raw_score"] == 1.0


def test_no_gold_answers_is_na_not_zero() -> None:
    result = score_vqa_accuracy({"output": "yellow"})
    assert result == {"metric_id": "vqa_accuracy", "status": "na", "na_reason": "no_gold"}
    assert "raw_score" not in result


def test_missing_output_is_na() -> None:
    result = score_vqa_accuracy({"gold_answers": ["yellow"] * 3})
    assert result["na_reason"] == "missing_output"


def test_empty_gold_answers_list_is_missing_evidence() -> None:
    result = score_vqa_accuracy({"output": "yellow", "gold_answers": []})
    assert result["na_reason"] == "missing_evidence"


def test_cancelled_beats_a_match() -> None:
    result = score_vqa_accuracy({"output": "yellow", "gold_answers": ["yellow"] * 10, "cancelled": True})
    assert result["na_reason"] == "cancelled"


def test_fills_capability_on_the_card() -> None:
    profile = build_profile(
        score_metrics(
            {
                "expected": "yes",
                "output": "blue",
                "grounding_judge": 1.0,
                "gold_answers": ["yellow"] * 10,
            }
        )
    )
    assert profile["capability"] == 0
