"""energy_per_query raw score: overage vs a declared kWh budget, lower-is-better.

Same shape as ``cost_per_query``/``latency_vs_budget``/``step_efficiency``:
pure instrumentation, no judge, no library. Declared-only by design:
this module does not measure energy itself. A caller who wants a real
number, not a hand-declared one, would wrap the call being scored with
something like CodeCarbon (or Vector Institute's own
``ai-impact-accounting``, which wraps CodeCarbon for model-lineage
training-footprint accounting (a different granularity, per-training-run
rather than per-query, so not directly reusable here) and write the
result onto ``energy_kwh`` before scoring, the same way a caller
supplies ``cost_usd`` today.

kWh, not gCO2e: "energy" and "carbon" are often named together, but
they are not the same declared quantity: converting kWh
to gCO2e needs a grid carbon-intensity factor that varies by region
and time, which this toolkit has no source for. kWh is the
instrumentation layer's native unit (CodeCarbon, ML CO2 Impact, and
most cloud provider APIs report kWh directly); a carbon-denominated
sibling metric, if ever wanted, would be a separate declared field,
not a silent conversion.
"""

from __future__ import annotations

from math import isfinite
from typing import Any

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "energy_per_query"
# Declared: clamp((energy - budget) / budget, 0, 1). Under budget is 0.0.


def score_energy_per_query(record: dict[str, Any]) -> dict[str, Any]:
    """Score how far this call's energy use ran past a declared kWh budget.

    Reads ``energy_kwh`` (>= 0) and ``energy_budget_kwh`` (> 0). Raw
    score is the overage fraction, clipped to ``[0, 1]``. On or under
    budget is 0.0. Grade uses the lower-is-better table. Missing or
    invalid numbers are N/A, never a silent 0.0.
    """
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("cancelled")

    energy, energy_ok = _finite_float(record, "energy_kwh")
    budget, budget_ok = _finite_float(record, "energy_budget_kwh")
    if not energy_ok or not budget_ok or energy is None or budget is None:
        return _na("missing_evidence")
    if energy < 0.0 or budget <= 0.0:
        return _na("missing_evidence")
    raw_score = min(1.0, max(0.0, (energy - budget) / budget))
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


def print_energy_per_query_examples() -> None:
    """Print how energy use and budget become a raw score and grade."""
    print("energy_per_query asks: did this call's energy use stay within the declared budget?")
    print("  raw_score = clamp((energy_kwh - energy_budget_kwh) / budget, 0, 1)")
    print("  On or under budget -> 0.0 (grade 4). Double the budget -> 1.0 (grade 0).")
    print("  Missing or invalid numbers -> na, not a silent 0.")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("0.0002 kWh call, 0.001 kWh budget (under).", {"energy_kwh": 0.0002, "energy_budget_kwh": 0.001}),
        ("Exactly on budget.", {"energy_kwh": 0.001, "energy_budget_kwh": 0.001}),
        ("25% over budget.", {"energy_kwh": 0.00025, "energy_budget_kwh": 0.0002}),
        ("Twice the budget (clipped).", {"energy_kwh": 0.003, "energy_budget_kwh": 0.001}),
        ("No budget on the record.", {"energy_kwh": 0.0002}),
        (
            "Case was cancelled (for example an audit stop).",
            {"energy_kwh": 0.0002, "energy_budget_kwh": 0.001, "cancelled": True},
        ),
    )
    for story, record in examples:
        result = score_energy_per_query(record)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_energy_per_query_examples()
