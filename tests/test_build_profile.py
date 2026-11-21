"""The eight-question card uses already-graded rows. It does not rescore."""

import pytest

from aixeval.scorecard import DIMENSIONS, build_profile


def test_task_success_fills_capability_only() -> None:
    results = [
        {"metric_id": "task_success", "status": "scored", "raw_score": 1.0, "grade": 4},
    ]
    profile = build_profile(results)
    assert profile["applied_hard_stops"] == ()
    assert profile["capability"] == 4
    assert profile["fairness"] is None
    assert all(profile[name] is None for name in DIMENSIONS if name != "capability")
    assert results[0]["raw_score"] == 1.0


def test_na_row_does_not_fill_a_dimension() -> None:
    profile = build_profile(
        [{"metric_id": "task_success", "status": "na", "na_reason": "no_gold"}],
    )
    assert profile["capability"] is None


def test_lowest_grade_wins_in_a_dimension() -> None:
    profile = build_profile(
        [
            {"metric_id": "task_success", "status": "scored", "raw_score": 1.0, "grade": 4},
            {"metric_id": "grounding", "status": "scored", "raw_score": 0.4, "grade": 2},
        ],
    )
    assert profile["capability"] == 2


def test_unknown_metric_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown metric_id"):
        build_profile([{"metric_id": "not_a_metric", "status": "scored", "grade": 4}])


def test_opt_in_metric_is_skipped_not_rejected() -> None:
    # toxicity/bertscore/meteor/clip_score are real, selectable, scorable
    # metrics, just deliberately outside the eight-question card's
    # starter set (METRIC_DIMENSION). A scored row for one must not crash
    # build_profile, and must not silently fill a dimension either.
    profile = build_profile(
        [
            {"metric_id": "task_success", "status": "scored", "raw_score": 1.0, "grade": 4},
            {"metric_id": "toxicity", "status": "scored", "raw_score": 0.1, "grade": 3},
        ],
    )
    assert profile["capability"] == 4
    assert all(profile[name] is None for name in DIMENSIONS if name != "capability")


def test_opt_in_metric_alone_still_produces_an_all_none_profile() -> None:
    profile = build_profile([{"metric_id": "toxicity", "status": "scored", "raw_score": 0.1, "grade": 3}])
    assert all(profile[name] is None for name in DIMENSIONS)
    assert profile["applied_hard_stops"] == ()


def test_scored_row_needs_a_grade() -> None:
    with pytest.raises(ValueError, match="missing an integer grade"):
        build_profile(
            [{"metric_id": "task_success", "status": "scored", "raw_score": 1.0}],
        )
