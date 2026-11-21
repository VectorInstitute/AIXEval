"""test_retest_stability raw score: variance across repeated scorings of a fixed input.

A different construct from ``consistency``, despite the similar-sounding
name: ``consistency`` asks whether the *model* behaves the same way
across two independent task attempts (declared ``run_a_score``/
``run_b_score``). This asks whether the *judge or scorer* returns the
same verdict when scoring the same, already-fixed output or trajectory
more than once: a check on the measurement, not on the model. Not by
re-verifying every individual verdict, but by measuring how much a
judge's score varies when asked to grade an unchanged input repeatedly.

Declared, not computed, same discipline as ``consistency``/
``modality_robustness``: this toolkit does not call a judge itself
here. A caller runs the same judge-backed metric N times on the same
(unchanged) input, e.g. ``judge/runner.py``'s ``run_grounding_judge``
called more than once on one record, and declares the resulting
scores as ``rerun_scores``. Grading reuses this toolkit's existing
clamp-to-budget-then-shared-cutoff-table idiom (the same one
``consistency``/``modality_robustness`` use), with its own 0.10
variance budget, lower-is-better.
"""

from __future__ import annotations

import statistics
from math import isfinite
from typing import Any

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "test_retest_stability"
VARIANCE_BUDGET = 0.10
MIN_RERUNS = 2


def score_test_retest_stability(record: dict[str, Any]) -> dict[str, Any]:
    """Score variance across repeated scorings of the same fixed input.

    Reads ``rerun_scores``: a list of at least two finite numbers on
    ``[0, 1]``, each one the same judge-backed metric's ``raw_score``
    from an independent call against the same, unchanged input. Raw
    score is ``clamp(population_variance(rerun_scores) / 0.10, 0, 1)``.
    Lower is better: ``0.0`` means every rerun agreed exactly. A
    missing, too-short, or invalid list is N/A, never a silent 0.0.
    """
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("cancelled")

    scores, ok = _rerun_scores(record)
    if not ok or scores is None:
        return _na("missing_evidence")

    variance = statistics.pvariance(scores)
    raw_score = min(1.0, variance / VARIANCE_BUDGET)
    return _scored(raw_score)


def _rerun_scores(record: dict[str, Any]) -> tuple[list[float] | None, bool]:
    """Return a validated score list, and whether the field type is usable."""
    if "rerun_scores" not in record or record["rerun_scores"] is None:
        return None, True
    value = record["rerun_scores"]
    if not isinstance(value, list) or len(value) < MIN_RERUNS:
        return None, False
    scores: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            return None, False
        number = float(item)
        if not isfinite(number) or number < 0.0 or number > 1.0:
            return None, False
        scores.append(number)
    return scores, True


def _scored(raw_score: float) -> dict[str, Any]:
    return apply_grade(
        {"metric_id": METRIC_ID, "status": "scored", "raw_score": raw_score},
        higher_is_better=False,
    )


def _na(reason: str) -> dict[str, Any]:
    return {"metric_id": METRIC_ID, "status": "na", "na_reason": require_na_reason(reason)}


def print_test_retest_stability_examples() -> None:
    """Print how a list of rerun scores becomes a raw score and grade."""
    print("test_retest_stability asks: does the SAME judge-backed metric agree with itself on a rerun?")
    print("  raw_score = clamp(population_variance(rerun_scores) / 0.10, 0, 1). Lower is better.")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("Three reruns, perfect agreement.", {"rerun_scores": [0.8, 0.8, 0.8]}),
        ("Three reruns, a little spread.", {"rerun_scores": [0.8, 0.85, 0.82]}),
        ("Three reruns, wide spread (clipped).", {"rerun_scores": [0.9, 0.5, 0.1]}),
        ("Only one rerun declared.", {"rerun_scores": [0.8]}),
        (
            "Case was cancelled (for example an audit stop).",
            {"rerun_scores": [0.9, 0.1], "cancelled": True},
        ),
    )
    for story, record in examples:
        result = score_test_retest_stability(record)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_test_retest_stability_examples()
