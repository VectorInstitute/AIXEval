"""score_cross_run_consistency: mean per-problem variance across a test set, or N/A."""

import pytest

from aixeval.cross_run_consistency import ProblemRuns, score_cross_run_consistency


def _bundle(raw_score: float | None, *, metric_id: str = "task_success", status: str = "scored") -> dict:
    if raw_score is None:
        row = {"metric_id": metric_id, "status": "na", "na_reason": "missing_evidence"}
    else:
        row = {"metric_id": metric_id, "status": status, "raw_score": raw_score}
    return {"scored_results": [row]}


def test_all_problems_perfectly_stable_is_zero_variance_grade_four() -> None:
    problems = [
        ProblemRuns("p1", [_bundle(1.0), _bundle(1.0), _bundle(1.0)]),
        ProblemRuns("p2", [_bundle(0.5), _bundle(0.5), _bundle(0.5)]),
    ]
    result = score_cross_run_consistency(problems, metric_id="task_success")
    assert result["status"] == "scored"
    assert result["mean_variance"] == 0.0
    assert result["raw_score"] == 0.0
    assert result["grade"] == 4
    assert result["unstable_share"] == 0.0
    assert result["unstable_problems"] == []
    assert result["problems_scored"] == 2
    assert result["excluded_problems"] == []


def test_matches_the_worked_example_from_spec() -> None:
    # Same shape as the spec's example: mostly low-variance, some higher -> grade 3.
    stable = [
        ProblemRuns(f"stable-{i}", [_bundle(1.0), _bundle(0.8)])  # pvariance([1.0, 0.8]) = 0.01
        for i in range(70)
    ]
    unstable = [
        ProblemRuns(f"unstable-{i}", [_bundle(1.0), _bundle(0.4)])  # pvariance([1.0, 0.4]) = 0.09
        for i in range(30)
    ]
    result = score_cross_run_consistency(stable + unstable, metric_id="task_success")
    expected_mean = (70 * 0.01 + 30 * 0.09) / 100
    assert result["mean_variance"] == pytest.approx(expected_mean)
    assert result["grade"] == 3


def test_unstable_share_and_list() -> None:
    problems = [
        ProblemRuns("stable", [_bundle(1.0), _bundle(1.0), _bundle(1.0)]),
        ProblemRuns("unstable", [_bundle(1.0), _bundle(0.0), _bundle(1.0)]),
    ]
    result = score_cross_run_consistency(problems, metric_id="task_success")
    assert result["unstable_share"] == 0.5
    assert [p["problem_id"] for p in result["unstable_problems"]] == ["unstable"]


def test_problem_with_too_few_runs_is_excluded() -> None:
    problems = [
        ProblemRuns("ok", [_bundle(1.0), _bundle(1.0)]),
        ProblemRuns("too_few", [_bundle(1.0)]),
    ]
    result = score_cross_run_consistency(problems, metric_id="task_success")
    assert result["problems_scored"] == 1
    assert result["excluded_problems"] == [{"problem_id": "too_few", "reason": "missing_evidence", "runs_available": 1}]


def test_na_rows_within_a_bundle_dont_count_as_usable_runs() -> None:
    problems = [
        ProblemRuns("p1", [_bundle(1.0), _bundle(1.0), _bundle(None)]),
    ]
    result = score_cross_run_consistency(problems, metric_id="task_success")
    # only 2 usable runs (the na row is skipped), still enough to score
    assert result["problems_scored"] == 1
    assert result["per_problem"][0]["runs"] == 2


def test_bundle_missing_the_metric_entirely_is_skipped() -> None:
    problems = [
        ProblemRuns(
            "p1",
            [
                _bundle(1.0),
                {"scored_results": [{"metric_id": "other_metric", "status": "scored", "raw_score": 0.5}]},
            ],
        ),
    ]
    result = score_cross_run_consistency(problems, metric_id="task_success")
    assert result["status"] == "na"
    assert result["excluded_problems"] == [{"problem_id": "p1", "reason": "missing_evidence", "runs_available": 1}]


def test_no_problem_has_enough_runs_is_na() -> None:
    problems = [ProblemRuns("p1", [_bundle(1.0)])]
    result = score_cross_run_consistency(problems, metric_id="task_success")
    assert result == {
        "status": "na",
        "na_reason": "missing_evidence",
        "excluded_problems": [{"problem_id": "p1", "reason": "missing_evidence", "runs_available": 1}],
    }


def test_empty_problem_list_is_na() -> None:
    result = score_cross_run_consistency([], metric_id="task_success")
    assert result["status"] == "na"
    assert result["excluded_problems"] == []


def test_variance_clips_to_one_at_the_budget() -> None:
    # pvariance([1.0, 0.0]) = 0.25, well past the 0.10 budget -> clamps to 1.0.
    problems = [ProblemRuns("p1", [_bundle(1.0), _bundle(0.0)])]
    result = score_cross_run_consistency(problems, metric_id="task_success")
    assert result["raw_score"] == 1.0
    assert result["grade"] == 0


def test_bootstrap_ci_is_none_with_fewer_than_two_scored_problems() -> None:
    problems = [ProblemRuns("p1", [_bundle(1.0), _bundle(1.0)])]
    result = score_cross_run_consistency(problems, metric_id="task_success")
    assert result["problems_scored"] == 1
    assert result["bootstrap_ci"] is None


def test_bootstrap_ci_is_present_and_reproducible_with_a_seed() -> None:
    problems = [
        ProblemRuns("p1", [_bundle(1.0), _bundle(1.0)]),
        ProblemRuns("p2", [_bundle(1.0), _bundle(0.0)]),
        ProblemRuns("p3", [_bundle(1.0), _bundle(0.5)]),
    ]
    result_a = score_cross_run_consistency(problems, metric_id="task_success", seed=42)
    result_b = score_cross_run_consistency(problems, metric_id="task_success", seed=42)
    assert result_a["bootstrap_ci"] is not None
    assert result_a["bootstrap_ci"] == result_b["bootstrap_ci"]
    low, high = result_a["bootstrap_ci"]
    assert 0.0 <= low <= high


def test_different_seeds_can_give_different_intervals() -> None:
    problems = [
        ProblemRuns("p1", [_bundle(1.0), _bundle(1.0)]),
        ProblemRuns("p2", [_bundle(1.0), _bundle(0.0)]),
        ProblemRuns("p3", [_bundle(1.0), _bundle(0.5)]),
    ]
    result_a = score_cross_run_consistency(problems, metric_id="task_success", seed=1, bootstrap_samples=50)
    result_b = score_cross_run_consistency(problems, metric_id="task_success", seed=2, bootstrap_samples=50)
    # Not asserting inequality (could coincide), just that both ran without error
    # and produced a valid interval.
    assert result_a["bootstrap_ci"] is not None
    assert result_b["bootstrap_ci"] is not None
