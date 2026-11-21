"""pairwise_preference grades a judge verdict on [0, 1], or N/A."""

from aixeval.scorecard import score_pairwise_preference


def test_output_clearly_better_is_one() -> None:
    result = score_pairwise_preference({"pairwise_preference_judge": 1.0})
    assert result == {
        "metric_id": "pairwise_preference",
        "status": "scored",
        "raw_score": 1.0,
        "grade": 4,
    }


def test_baseline_clearly_better_is_zero() -> None:
    result = score_pairwise_preference({"pairwise_preference_judge": 0.0})
    assert result["status"] == "scored"
    assert result["raw_score"] == 0.0
    assert result["grade"] == 0


def test_partial_verdict_keeps_the_raw_score() -> None:
    result = score_pairwise_preference({"pairwise_preference_judge": 0.6})
    assert result["raw_score"] == 0.6
    assert result["grade"] == 3


def test_integer_one_is_accepted() -> None:
    assert score_pairwise_preference({"pairwise_preference_judge": 1})["raw_score"] == 1.0


def test_no_verdict_no_baseline_is_na_not_zero() -> None:
    result = score_pairwise_preference({"output": "Paris."})
    assert result == {"metric_id": "pairwise_preference", "status": "na", "na_reason": "missing_evidence"}
    assert "raw_score" not in result
    assert "grade" not in result


def test_bool_verdict_is_na() -> None:
    assert score_pairwise_preference({"pairwise_preference_judge": True})["na_reason"] == "missing_evidence"


def test_out_of_range_is_na() -> None:
    assert score_pairwise_preference({"pairwise_preference_judge": 1.2})["na_reason"] == "missing_evidence"
    assert score_pairwise_preference({"pairwise_preference_judge": -0.1})["na_reason"] == "missing_evidence"


def test_cancelled_beats_a_verdict() -> None:
    result = score_pairwise_preference({"pairwise_preference_judge": 1.0, "cancelled": True})
    assert result["status"] == "na"
    assert result["na_reason"] == "cancelled"


def test_injected_judge_uses_output_and_baseline() -> None:
    seen: list[tuple[str, str]] = []

    def judge(output: str, baseline_output: str) -> float:
        seen.append((output, baseline_output))
        return 0.7

    result = score_pairwise_preference(
        {"output": "Paris.", "baseline_output": "The capital of France is Paris."},
        judge=judge,
    )
    assert seen == [("Paris.", "The capital of France is Paris.")]
    assert result["raw_score"] == 0.7
    assert result["grade"] == 3


def test_injected_judge_needs_a_baseline() -> None:
    result = score_pairwise_preference({"output": "Paris."}, judge=lambda _o, _b: 1.0)
    assert result["na_reason"] == "no_gold"


def test_injected_judge_needs_output() -> None:
    result = score_pairwise_preference({"baseline_output": "Paris."}, judge=lambda _o, _b: 1.0)
    assert result["na_reason"] == "missing_output"


def test_stored_verdict_wins_over_injected_judge() -> None:
    result = score_pairwise_preference(
        {"pairwise_preference_judge": 0.0, "output": "x", "baseline_output": "y"},
        judge=lambda _o, _b: 1.0,
    )
    assert result["raw_score"] == 0.0
