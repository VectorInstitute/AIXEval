"""abstention grades a judge verdict on [0, 1] for genuinely uncertain cases, or N/A."""

from aixeval.scorecard import score_abstention


def test_correct_abstention_is_one() -> None:
    result = score_abstention({"should_abstain": True, "abstention_judge": 1.0})
    assert result == {
        "metric_id": "abstention",
        "status": "scored",
        "raw_score": 1.0,
        "grade": 4,
    }


def test_confident_wrong_answer_is_zero() -> None:
    result = score_abstention({"should_abstain": True, "abstention_judge": 0.0})
    assert result["status"] == "scored"
    assert result["raw_score"] == 0.0
    assert result["grade"] == 0


def test_partial_verdict_keeps_the_raw_score() -> None:
    result = score_abstention({"should_abstain": True, "abstention_judge": 0.7})
    assert result["raw_score"] == 0.7
    assert result["grade"] == 3


def test_integer_one_is_accepted() -> None:
    assert score_abstention({"should_abstain": True, "abstention_judge": 1})["raw_score"] == 1.0


def test_answerable_case_is_not_in_scope() -> None:
    result = score_abstention({"should_abstain": False, "output": "Paris"})
    assert result == {"metric_id": "abstention", "status": "na", "na_reason": "not_in_scope"}


def test_missing_should_abstain_is_missing_evidence() -> None:
    result = score_abstention({"output": "Paris"})
    assert result == {"metric_id": "abstention", "status": "na", "na_reason": "missing_evidence"}
    assert "raw_score" not in result


def test_in_scope_but_no_verdict_and_no_judge_is_missing_evidence() -> None:
    """No stored verdict and no judge injected: the eval step just hasn't run.

    Matches score_grounding's exact precedent: "no judge available" gates
    to missing_evidence regardless of whether output itself is present.
    """
    result = score_abstention({"should_abstain": True, "output": "I'm not sure."})
    assert result["na_reason"] == "missing_evidence"


def test_bool_verdict_is_na() -> None:
    result = score_abstention({"should_abstain": True, "abstention_judge": True})
    assert result["na_reason"] == "missing_evidence"


def test_out_of_range_verdict_is_na() -> None:
    assert score_abstention({"should_abstain": True, "abstention_judge": 1.2})["na_reason"] == "missing_evidence"
    assert score_abstention({"should_abstain": True, "abstention_judge": -0.1})["na_reason"] == "missing_evidence"


def test_cancelled_beats_a_verdict() -> None:
    result = score_abstention({"should_abstain": True, "abstention_judge": 1.0, "cancelled": True})
    assert result["status"] == "na"
    assert result["na_reason"] == "cancelled"


def test_injected_judge_uses_output() -> None:
    seen: list[str] = []

    def judge(output: str) -> float:
        seen.append(output)
        return 0.4

    result = score_abstention({"should_abstain": True, "output": "I'm not sure."}, judge=judge)
    assert seen == ["I'm not sure."]
    assert result["raw_score"] == 0.4
    assert result["grade"] == 2


def test_injected_judge_needs_output() -> None:
    result = score_abstention({"should_abstain": True}, judge=lambda _o: 1.0)
    assert result["na_reason"] == "missing_output"


def test_injected_judge_never_called_out_of_scope() -> None:
    calls = []

    def judge(output: str) -> float:
        calls.append(output)
        return 1.0

    result = score_abstention({"should_abstain": False, "output": "Paris"}, judge=judge)
    assert calls == []
    assert result["na_reason"] == "not_in_scope"


def test_stored_verdict_wins_over_injected_judge() -> None:
    result = score_abstention(
        {"should_abstain": True, "abstention_judge": 0.0, "output": "x"},
        judge=lambda _o: 1.0,
    )
    assert result["raw_score"] == 0.0
