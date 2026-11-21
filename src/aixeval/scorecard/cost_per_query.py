"""cost_per_query raw score: overage vs a declared dollar budget, lower-is-better.

Same shape as ``latency_vs_budget``/``step_efficiency``: pure
instrumentation, no judge, no library. A genuinely per-record fact
(what this one call cost), unlike Throughput or Memory footprint,
which are rates or static deployment facts, not properties of a
single record, left
out of this toolkit for that reason, not overlooked.
"""

from __future__ import annotations

from math import isfinite
from typing import Any

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "cost_per_query"
# Declared: clamp((cost - budget) / budget, 0, 1). Under budget is 0.0.


def score_cost_per_query(record: dict[str, Any]) -> dict[str, Any]:
    """Score how far this call's cost ran past a declared dollar budget.

    Reads ``cost_usd`` (>= 0) and ``cost_budget_usd`` (> 0). Raw score
    is the overage fraction, clipped to ``[0, 1]``. On or under budget
    is 0.0. Grade uses the lower-is-better table. Missing or invalid
    numbers are N/A, never a silent 0.0.
    """
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("cancelled")

    cost, cost_ok = _finite_float(record, "cost_usd")
    budget, budget_ok = _finite_float(record, "cost_budget_usd")
    if not cost_ok or not budget_ok or cost is None or budget is None:
        return _na("missing_evidence")
    if cost < 0.0 or budget <= 0.0:
        return _na("missing_evidence")
    raw_score = min(1.0, max(0.0, (cost - budget) / budget))
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


def print_cost_per_query_examples() -> None:
    """Print how cost and budget become a raw score and grade."""
    print("cost_per_query asks: did this call's cost stay within the declared budget?")
    print("  raw_score = clamp((cost_usd - cost_budget_usd) / budget, 0, 1)")
    print("  On or under budget -> 0.0 (grade 4). Double the budget -> 1.0 (grade 0).")
    print("  Missing or invalid numbers -> na, not a silent 0.")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("$0.003 call, $0.01 budget (under).", {"cost_usd": 0.003, "cost_budget_usd": 0.01}),
        ("Exactly on budget.", {"cost_usd": 0.01, "cost_budget_usd": 0.01}),
        ("20% over budget.", {"cost_usd": 0.012, "cost_budget_usd": 0.01}),
        ("Twice the budget (clipped).", {"cost_usd": 0.03, "cost_budget_usd": 0.01}),
        ("No budget on the record.", {"cost_usd": 0.003}),
        (
            "Case was cancelled (for example an audit stop).",
            {"cost_usd": 0.003, "cost_budget_usd": 0.01, "cancelled": True},
        ),
    )
    for story, record in examples:
        result = score_cost_per_query(record)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_cost_per_query_examples()
