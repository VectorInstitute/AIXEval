"""step_efficiency raw score: extra steps vs a declared path, lower-is-better."""

from __future__ import annotations

from typing import Any

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "step_efficiency"
# Declared: clamp((steps - step_budget) / step_budget, 0, 1). Under budget is 0.0.


def score_step_efficiency(record: dict[str, Any]) -> dict[str, Any]:
    """Score how far the run ran past a declared step budget.

    Reads ``steps`` (int >= 0) and ``step_budget`` (int > 0). Raw score
    is the overage fraction, clipped to ``[0, 1]``. On or under budget
    is 0.0. Grade uses the lower-is-better table. Missing or invalid
    counts are N/A, never a silent 0.0.
    """
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("cancelled")

    steps, steps_ok = _count(record, "steps")
    budget, budget_ok = _count(record, "step_budget")
    if not steps_ok or not budget_ok or steps is None or budget is None:
        return _na("missing_evidence")
    if steps < 0 or budget <= 0:
        return _na("missing_evidence")
    raw_score = min(1.0, max(0.0, (steps - budget) / budget))
    return _scored(raw_score)


def _count(record: dict[str, Any], key: str) -> tuple[int | None, bool]:
    """Return an int count and whether the field type is usable."""
    if key not in record or record[key] is None:
        return None, True
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, int):
        return None, False
    return value, True


def _scored(raw_score: float) -> dict[str, Any]:
    return apply_grade(
        {"metric_id": METRIC_ID, "status": "scored", "raw_score": raw_score},
        higher_is_better=False,
    )


def _na(reason: str) -> dict[str, Any]:
    return {"metric_id": METRIC_ID, "status": "na", "na_reason": require_na_reason(reason)}


def print_step_efficiency_examples() -> None:
    """Print how step counts become a raw score and grade."""
    print("step_efficiency asks: did the run use about as many steps as the declared path?")
    print("  raw_score = clamp((steps - step_budget) / step_budget, 0, 1)")
    print("  On or under budget -> 0.0 (grade 4). Double the path -> 1.0 (grade 0).")
    print("  Missing or non-int counts -> na, not a silent 0.")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("8 steps, budget 10 (under).", {"steps": 8, "step_budget": 10}),
        ("Exactly on budget.", {"steps": 10, "step_budget": 10}),
        ("12 steps, budget 10 (20% over).", {"steps": 12, "step_budget": 10}),
        ("Twice the path (clipped).", {"steps": 21, "step_budget": 8}),
        ("No budget on the record.", {"steps": 8}),
        (
            "Case was cancelled (for example an audit stop).",
            {"steps": 8, "step_budget": 10, "cancelled": True},
        ),
    )
    for story, record in examples:
        result = score_step_efficiency(record)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_step_efficiency_examples()
