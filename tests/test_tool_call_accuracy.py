"""tool_call_accuracy grades a stored or injected verdict on [0, 1], or N/A.

The verdict can come from either mechanism (deterministic check or LLM
judge, see aixeval.judge.runner): this scorer only ever grades
the verdict already on the record, or injects a judge directly. It
never compares tool calls itself.
"""

from aixeval.scorecard import score_tool_call_accuracy


def test_fully_correct_is_one() -> None:
    result = score_tool_call_accuracy({"tool_call_accuracy_judge": 1.0})
    assert result == {
        "metric_id": "tool_call_accuracy",
        "status": "scored",
        "raw_score": 1.0,
        "grade": 4,
    }


def test_fully_wrong_is_zero() -> None:
    result = score_tool_call_accuracy({"tool_call_accuracy_judge": 0.0})
    assert result["status"] == "scored"
    assert result["raw_score"] == 0.0
    assert result["grade"] == 0


def test_partial_verdict_keeps_the_raw_score() -> None:
    result = score_tool_call_accuracy({"tool_call_accuracy_judge": 0.7})
    assert result["raw_score"] == 0.7
    assert result["grade"] == 3


def test_integer_one_is_accepted() -> None:
    assert score_tool_call_accuracy({"tool_call_accuracy_judge": 1})["raw_score"] == 1.0


def test_no_verdict_no_judge_is_na_not_zero() -> None:
    result = score_tool_call_accuracy(
        {"output": "68F", "trajectory": "1. get_weather -- Paris", "gold_tool_plan": "call get_weather"}
    )
    assert result == {"metric_id": "tool_call_accuracy", "status": "na", "na_reason": "missing_evidence"}
    assert "raw_score" not in result
    assert "grade" not in result


def test_bool_verdict_is_na() -> None:
    assert score_tool_call_accuracy({"tool_call_accuracy_judge": True})["na_reason"] == "missing_evidence"


def test_out_of_range_is_na() -> None:
    assert score_tool_call_accuracy({"tool_call_accuracy_judge": 1.2})["na_reason"] == "missing_evidence"
    assert score_tool_call_accuracy({"tool_call_accuracy_judge": -0.1})["na_reason"] == "missing_evidence"


def test_cancelled_beats_a_verdict() -> None:
    result = score_tool_call_accuracy({"tool_call_accuracy_judge": 1.0, "cancelled": True})
    assert result["status"] == "na"
    assert result["na_reason"] == "cancelled"


def test_injected_judge_uses_trajectory_gold_plan_and_output() -> None:
    seen: list[tuple[str, str, str]] = []

    def judge(trajectory: str, gold_tool_plan: str, output: str) -> float:
        seen.append((trajectory, gold_tool_plan, output))
        return 0.4

    result = score_tool_call_accuracy(
        {"trajectory": "1. get_weather -- Paris", "gold_tool_plan": "call get_weather", "output": "68F"},
        judge=judge,
    )
    assert seen == [("1. get_weather -- Paris", "call get_weather", "68F")]
    assert result["raw_score"] == 0.4
    assert result["grade"] == 2


def test_injected_judge_needs_a_trajectory() -> None:
    result = score_tool_call_accuracy(
        {"gold_tool_plan": "call get_weather", "output": "68F"}, judge=lambda _t, _g, _o: 1.0
    )
    assert result["na_reason"] == "missing_evidence"


def test_injected_judge_needs_gold_tool_plan() -> None:
    result = score_tool_call_accuracy(
        {"trajectory": "1. get_weather -- Paris", "output": "68F"}, judge=lambda _t, _g, _o: 1.0
    )
    assert result["na_reason"] == "no_gold"


def test_injected_judge_needs_output() -> None:
    result = score_tool_call_accuracy(
        {"trajectory": "1. get_weather -- Paris", "gold_tool_plan": "call get_weather"},
        judge=lambda _t, _g, _o: 1.0,
    )
    assert result["na_reason"] == "missing_output"


def test_stored_verdict_wins_over_injected_judge() -> None:
    result = score_tool_call_accuracy(
        {
            "tool_call_accuracy_judge": 0.0,
            "trajectory": "x",
            "gold_tool_plan": "y",
            "output": "z",
        },
        judge=lambda _t, _g, _o: 1.0,
    )
    assert result["raw_score"] == 0.0
