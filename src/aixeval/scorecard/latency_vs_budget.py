"""latency_vs_budget raw score: overage vs a declared budget, lower-is-better."""

from __future__ import annotations

from math import isfinite
from typing import Any

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "latency_vs_budget"
# Declared: clamp((latency - budget) / budget, 0, 1). Under budget is 0.0.


def score_latency_vs_budget(record: dict[str, Any]) -> dict[str, Any]:
    """Score how far wall time ran past a declared budget.

    Reads ``latency_sec`` (>= 0) and ``latency_budget_sec`` (> 0).
    Raw score is the overage fraction, clipped to ``[0, 1]``. On or
    under budget is 0.0. Grade uses the lower-is-better table.
    Missing or invalid numbers are N/A, never a silent 0.0.
    """
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("cancelled")

    latency, latency_ok = _finite_float(record, "latency_sec")
    budget, budget_ok = _finite_float(record, "latency_budget_sec")
    if not latency_ok or not budget_ok or latency is None or budget is None:
        return _na("missing_evidence")
    if latency < 0.0 or budget <= 0.0:
        return _na("missing_evidence")
    raw_score = min(1.0, max(0.0, (latency - budget) / budget))
    return _scored(raw_score)


def _finite_float(record: dict[str, Any], key: str) -> tuple[float | None, bool]:
    """Return a finite float and whether the field type is usable."""
    if key not in record or record[key] is None:
        return None, True
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None, False
    number = float(value)
    if not isfinite(number):
        return None, False
    return number, True


def _scored(raw_score: float) -> dict[str, Any]:
    return apply_grade(
        {"metric_id": METRIC_ID, "status": "scored", "raw_score": raw_score},
        higher_is_better=False,
    )


def _na(reason: str) -> dict[str, Any]:
    return {"metric_id": METRIC_ID, "status": "na", "na_reason": require_na_reason(reason)}


def print_latency_vs_budget_examples() -> None:
    """Print how latency and budget become a raw score and grade."""
    print("latency_vs_budget asks: did elapsed time stay within the declared budget?")
    print("  raw_score = clamp((latency_sec - latency_budget_sec) / budget, 0, 1)")
    print("  On or under budget -> 0.0 (grade 4). Double the budget -> 1.0 (grade 0).")
    print("  Missing or invalid numbers -> na, not a silent 0.")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("31s run, 60s budget (under).", {"latency_sec": 31.0, "latency_budget_sec": 60.0}),
        ("Exactly on budget.", {"latency_sec": 60.0, "latency_budget_sec": 60.0}),
        ("20% over budget.", {"latency_sec": 72.0, "latency_budget_sec": 60.0}),
        ("Twice the budget (clipped).", {"latency_sec": 176.0, "latency_budget_sec": 60.0}),
        ("No budget on the record.", {"latency_sec": 31.0}),
        (
            "Case was cancelled (for example an audit stop).",
            {"latency_sec": 31.0, "latency_budget_sec": 60.0, "cancelled": True},
        ),
    )
    for story, record in examples:
        result = score_latency_vs_budget(record)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_latency_vs_budget_examples()
