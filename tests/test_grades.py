"""Higher-is-better raw scores map to a declared 0-4 grade."""

import pytest

from aixeval.scorecard import (
    HIGHER_IS_BETTER_CUTOFFS,
    LOWER_IS_BETTER_CUTOFFS,
    apply_grade,
    grade_for_raw_score,
    label_for_band,
)


def test_cutoffs_are_the_declared_policy_table() -> None:
    assert HIGHER_IS_BETTER_CUTOFFS == (0.20, 0.40, 0.60, 0.80)
    assert LOWER_IS_BETTER_CUTOFFS == HIGHER_IS_BETTER_CUTOFFS


@pytest.mark.parametrize(
    ("raw_score", "grade"),
    [
        (0.0, 0),
        (0.199999, 0),
        (0.20, 1),
        (0.399999, 1),
        (0.40, 2),
        (0.60, 3),
        (0.79, 3),
        (0.80, 4),
        (1.0, 4),
    ],
)
def test_edges_of_higher_is_better(raw_score: float, grade: int) -> None:
    assert grade_for_raw_score(raw_score) == grade


@pytest.mark.parametrize(
    ("raw_score", "grade"),
    [
        (0.0, 4),
        (0.20, 3),
        (0.40, 2),
        (0.60, 1),
        (0.80, 0),
        (1.0, 0),
    ],
)
def test_edges_of_lower_is_better(raw_score: float, grade: int) -> None:
    assert grade_for_raw_score(raw_score, higher_is_better=False) == grade


def test_lower_is_better_apply_grade() -> None:
    scored = apply_grade(
        {"metric_id": "unsafe_action", "status": "scored", "raw_score": 0.0},
        higher_is_better=False,
    )
    assert scored["raw_score"] == 0.0
    assert scored["grade"] == 4


def test_out_of_range_raw_score_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside 0-1"):
        grade_for_raw_score(-0.01)
    with pytest.raises(ValueError, match="outside 0-1"):
        grade_for_raw_score(1.01)


def test_apply_grade_keeps_raw_score_and_skips_na() -> None:
    scored = apply_grade({"metric_id": "task_success", "status": "scored", "raw_score": 1.0})
    assert scored["raw_score"] == 1.0
    assert scored["grade"] == 4
    assert label_for_band(scored["grade"]) == "Excellent"

    na = apply_grade({"metric_id": "task_success", "status": "na", "na_reason": "no_gold"})
    assert "grade" not in na
    assert "raw_score" not in na
