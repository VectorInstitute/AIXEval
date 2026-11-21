"""pass_at_k_agent is 1.0 (solved_within_budget) or 0.0 (not), or N/A."""

from aixeval.scorecard import build_profile, score_metrics, score_pass_at_k_agent


def test_solved_within_budget_is_one() -> None:
    result = score_pass_at_k_agent({"attempt_budget": 3, "solved_within_budget": True})
    assert result == {
        "metric_id": "pass_at_k_agent",
        "status": "scored",
        "raw_score": 1.0,
        "grade": 4,
    }


def test_not_solved_is_zero() -> None:
    result = score_pass_at_k_agent({"attempt_budget": 3, "solved_within_budget": False})
    assert result["raw_score"] == 0.0
    assert result["grade"] == 0


def test_missing_attempt_budget_is_missing_evidence() -> None:
    result = score_pass_at_k_agent({"solved_within_budget": True})
    assert result["na_reason"] == "missing_evidence"


def test_missing_solved_flag_is_missing_evidence() -> None:
    result = score_pass_at_k_agent({"attempt_budget": 3})
    assert result["na_reason"] == "missing_evidence"


def test_zero_or_negative_budget_is_missing_evidence() -> None:
    assert score_pass_at_k_agent({"attempt_budget": 0, "solved_within_budget": True})["na_reason"] == (
        "missing_evidence"
    )
    assert score_pass_at_k_agent({"attempt_budget": -1, "solved_within_budget": True})["na_reason"] == (
        "missing_evidence"
    )


def test_bool_attempt_budget_is_missing_evidence() -> None:
    result = score_pass_at_k_agent({"attempt_budget": True, "solved_within_budget": True})
    assert result["na_reason"] == "missing_evidence"


def test_cancelled_beats_a_solve() -> None:
    result = score_pass_at_k_agent({"attempt_budget": 3, "solved_within_budget": True, "cancelled": True})
    assert result["na_reason"] == "cancelled"


def test_fills_capability_on_the_card() -> None:
    profile = build_profile(
        score_metrics(
            {
                "expected": "yes",
                "output": "yes",
                "grounding_judge": 1.0,
                "attempt_budget": 3,
                "solved_within_budget": False,
            }
        )
    )
    assert profile["capability"] == 0
