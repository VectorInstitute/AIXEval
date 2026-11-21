"""consistency raw score: spread between two repeated runs of the same task.

Declared, not computed: this toolkit does not re-run the task. The
harness runs the same task twice under identical conditions, scores
each run on some already-chosen [0, 1] metric, and declares both
(``run_a_score``, ``run_b_score``). This module only computes the
spread between them.

Grading: there's no pinned budget for run-to-run variance for this
construct, so this module reuses the same declared 0.30 ceiling as
``modality_robustness`` and the
shared ``apply_grade()`` cutoff table, rather than inventing a second
budget, a declared simplification, flagged here rather than
silently diverging. Lower is better.
"""

from __future__ import annotations

from math import isfinite
from typing import Any

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "consistency"
SPREAD_BUDGET = 0.30


def score_consistency(record: dict[str, Any]) -> dict[str, Any]:
    """Score how far apart two repeated runs of the same task landed.

    Reads ``run_a_score`` and ``run_b_score`` (both on ``[0, 1]``).
    Raw score is ``clamp(abs(run_a_score - run_b_score) / 0.30, 0, 1)``.
    Identical runs are 0.0. Lower is better. Missing or invalid numbers
    are N/A, never a silent 0.0.
    """
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("cancelled")

    run_a, run_a_ok = _optional_unit(record, "run_a_score")
    run_b, run_b_ok = _optional_unit(record, "run_b_score")
    if not run_a_ok or not run_b_ok or run_a is None or run_b is None:
        return _na("missing_evidence")

    spread = abs(run_a - run_b)
    raw_score = min(1.0, spread / SPREAD_BUDGET)
    return _scored(raw_score)


def _optional_unit(record: dict[str, Any], key: str) -> tuple[float | None, bool]:
    """Return a finite float on [0, 1] and whether the field type is usable."""
    if key not in record or record[key] is None:
        return None, True
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None, False
    number = float(value)
    if not isfinite(number) or number < 0.0 or number > 1.0:
        return None, False
    return number, True


def _scored(raw_score: float) -> dict[str, Any]:
    return apply_grade(
        {"metric_id": METRIC_ID, "status": "scored", "raw_score": raw_score},
        higher_is_better=False,
    )


def _na(reason: str) -> dict[str, Any]:
    return {"metric_id": METRIC_ID, "status": "na", "na_reason": require_na_reason(reason)}


def print_consistency_examples() -> None:
    """Print how two repeated-run scores become a raw score and grade."""
    print("consistency asks: how far apart did two repeated runs of the same task land?")
    print("  raw_score = clamp(abs(run_a_score - run_b_score) / 0.30, 0, 1). Lower is better.")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("Two identical runs.", {"run_a_score": 0.8, "run_b_score": 0.8}),
        ("A 15-point spread between runs.", {"run_a_score": 0.9, "run_b_score": 0.75}),
        ("A 50-point spread (clipped).", {"run_a_score": 0.9, "run_b_score": 0.4}),
        ("Only one run declared.", {"run_a_score": 0.9}),
        (
            "Case was cancelled (for example an audit stop).",
            {"run_a_score": 0.9, "run_b_score": 0.4, "cancelled": True},
        ),
    )
    for story, record in examples:
        result = score_consistency(record)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_consistency_examples()
