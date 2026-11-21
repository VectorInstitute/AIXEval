"""pass_hat_k_agent is comb(c, k) / comb(n, k), k = min(5, n), or N/A."""

import pytest

from aixeval.scorecard import build_profile, score_metrics, score_pass_hat_k_agent


def test_all_runs_succeed_is_one_and_grade_four() -> None:
    result = score_pass_hat_k_agent({"pass_hat_k_attempts": 5, "pass_hat_k_successes": 5})
    assert result == {
        "metric_id": "pass_hat_k_agent",
        "status": "scored",
        "raw_score": 1.0,
        "grade": 4,
    }


def test_four_of_five_is_zero() -> None:
    result = score_pass_hat_k_agent({"pass_hat_k_attempts": 5, "pass_hat_k_successes": 4})
    assert result["raw_score"] == 0.0
    assert result["grade"] == 0


def test_no_successes_is_zero() -> None:
    result = score_pass_hat_k_agent({"pass_hat_k_attempts": 5, "pass_hat_k_successes": 0})
    assert result["raw_score"] == 0.0


def test_k_caps_at_fewer_declared_attempts() -> None:
    # Only 3 runs declared: k = min(5, 3) = 3, so 3/3 successes -> comb(3,3)/comb(3,3).
    result = score_pass_hat_k_agent({"pass_hat_k_attempts": 3, "pass_hat_k_successes": 3})
    assert result["raw_score"] == 1.0


def test_matches_the_worked_example_with_seven_attempts() -> None:
    # k = min(5, 7) = 5. comb(6, 5) / comb(7, 5) = 6 / 21.
    result = score_pass_hat_k_agent({"pass_hat_k_attempts": 7, "pass_hat_k_successes": 6})
    assert result["raw_score"] == pytest.approx(6 / 21)


def test_missing_or_invalid_is_na() -> None:
    assert score_pass_hat_k_agent({"pass_hat_k_successes": 3})["na_reason"] == "missing_evidence"
    assert score_pass_hat_k_agent({"pass_hat_k_attempts": 5})["na_reason"] == "missing_evidence"
    assert (
        score_pass_hat_k_agent({"pass_hat_k_attempts": 0, "pass_hat_k_successes": 0})["na_reason"] == "missing_evidence"
    )
    assert (
        score_pass_hat_k_agent({"pass_hat_k_attempts": -1, "pass_hat_k_successes": 0})["na_reason"]
        == "missing_evidence"
    )


def test_successes_exceeding_attempts_is_na() -> None:
    result = score_pass_hat_k_agent({"pass_hat_k_attempts": 3, "pass_hat_k_successes": 5})
    assert result["na_reason"] == "missing_evidence"


def test_bool_is_missing_evidence() -> None:
    assert (
        score_pass_hat_k_agent({"pass_hat_k_attempts": True, "pass_hat_k_successes": 1})["na_reason"]
        == "missing_evidence"
    )
    assert (
        score_pass_hat_k_agent({"pass_hat_k_attempts": 5, "pass_hat_k_successes": True})["na_reason"]
        == "missing_evidence"
    )


def test_cancelled_beats_a_clean_success() -> None:
    result = score_pass_hat_k_agent({"pass_hat_k_attempts": 5, "pass_hat_k_successes": 5, "cancelled": True})
    assert result["na_reason"] == "cancelled"


def test_fills_capability_on_the_card() -> None:
    profile = build_profile(
        score_metrics(
            {
                "expected": "yes",
                "output": "yes",
                "pass_hat_k_attempts": 5,
                "pass_hat_k_successes": 4,
            }
        )
    )
    assert profile["capability"] == 0
