"""grounding grades a judge verdict on [0, 1], or N/A."""

from aixeval.scorecard import score_grounding


def test_full_support_is_one() -> None:
    result = score_grounding({"grounding_judge": 1.0})
    assert result == {
        "metric_id": "grounding",
        "status": "scored",
        "raw_score": 1.0,
        "grade": 4,
    }


def test_unsupported_is_zero() -> None:
    result = score_grounding({"grounding_judge": 0.0})
    assert result["status"] == "scored"
    assert result["raw_score"] == 0.0
    assert result["grade"] == 0


def test_partial_verdict_keeps_the_raw_score() -> None:
    result = score_grounding({"grounding_judge": 0.7})
    assert result["raw_score"] == 0.7
    assert result["grade"] == 3


def test_integer_one_is_accepted() -> None:
    assert score_grounding({"grounding_judge": 1})["raw_score"] == 1.0


def test_no_verdict_is_na_not_zero() -> None:
    result = score_grounding({"output": "yes", "evidence": "yes"})
    assert result == {"metric_id": "grounding", "status": "na", "na_reason": "missing_evidence"}
    assert "raw_score" not in result
    assert "grade" not in result


def test_bool_verdict_is_na() -> None:
    assert score_grounding({"grounding_judge": True})["na_reason"] == "missing_evidence"


def test_out_of_range_is_na() -> None:
    assert score_grounding({"grounding_judge": 1.2})["na_reason"] == "missing_evidence"
    assert score_grounding({"grounding_judge": -0.1})["na_reason"] == "missing_evidence"


def test_cancelled_beats_a_verdict() -> None:
    result = score_grounding({"grounding_judge": 1.0, "cancelled": True})
    assert result["status"] == "na"
    assert result["na_reason"] == "cancelled"


def test_injected_judge_uses_evidence_and_output() -> None:
    seen: list[tuple[str, str]] = []

    def judge(evidence: str, output: str) -> float:
        seen.append((evidence, output))
        return 0.4

    result = score_grounding({"evidence": "the chart", "output": "it rose"}, judge=judge)
    assert seen == [("the chart", "it rose")]
    assert result["raw_score"] == 0.4
    assert result["grade"] == 2


def test_injected_judge_needs_output() -> None:
    result = score_grounding({"evidence": "the chart"}, judge=lambda _e, _o: 1.0)
    assert result["na_reason"] == "missing_output"


def test_stored_verdict_wins_over_injected_judge() -> None:
    result = score_grounding(
        {"grounding_judge": 0.0, "evidence": "x", "output": "x"},
        judge=lambda _e, _o: 1.0,
    )
    assert result["raw_score"] == 0.0
