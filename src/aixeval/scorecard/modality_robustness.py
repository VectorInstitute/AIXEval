"""modality_robustness raw score: performance degradation under perturbation.

Declared, not computed: this toolkit does not run the perturbation
itself (adding image noise, audio dropout, etc.). The harness runs
the same task twice (once clean, once perturbed), scores each
condition on some already-chosen [0, 1] metric, and declares both
results (``clean_score``, ``perturbed_score``). This module only
computes the degradation between them.

Grading: this construct is naturally a budget-ratio normalization
with ceiling 0.30, a different banding formula
(``band = 4 - floor(4v/B)``) from the fixed H/L cutoff tables
every other metric here uses. For consistency with every sibling
lower-is-better metric (``cost_per_query``, ``energy_per_query``,
``latency_vs_budget``), this module reuses the existing
``clamp(value / budget, 0, 1)`` ratio-to-budget pattern and the
shared ``apply_grade()`` cutoff table instead of introducing a second,
one-off grading formula, a declared simplification, flagged here
rather than silently diverging.
"""

from __future__ import annotations

from math import isfinite
from typing import Any

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "modality_robustness"
# The declared budget ceiling for this metric.
DEGRADATION_BUDGET = 0.30


def score_modality_robustness(record: dict[str, Any]) -> dict[str, Any]:
    """Score performance degradation between a clean and a perturbed run.

    Reads ``clean_score`` and ``perturbed_score`` (both on ``[0, 1]``).
    Raw score is ``clamp((clean_score - perturbed_score) / 0.30, 0, 1)``:
    no degradation (perturbed >= clean) is 0.0. Lower is better.
    Missing or invalid numbers are N/A, never a silent 0.0.
    """
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("cancelled")

    clean, clean_ok = _optional_unit(record, "clean_score")
    perturbed, perturbed_ok = _optional_unit(record, "perturbed_score")
    if not clean_ok or not perturbed_ok or clean is None or perturbed is None:
        return _na("missing_evidence")

    degradation = max(0.0, clean - perturbed)
    raw_score = min(1.0, degradation / DEGRADATION_BUDGET)
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


def print_modality_robustness_examples() -> None:
    """Print how a clean/perturbed score pair becomes a raw score and grade."""
    print("modality_robustness asks: how much did performance drop under perturbation?")
    print("  raw_score = clamp((clean_score - perturbed_score) / 0.30, 0, 1). Lower is better.")
    print("  No drop (or an improvement) -> 0.0 (grade 4). A 30%+ drop -> 1.0 (grade 0).")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("No degradation under perturbation.", {"clean_score": 0.9, "perturbed_score": 0.9}),
        ("Perturbed score actually higher (no degradation).", {"clean_score": 0.8, "perturbed_score": 0.85}),
        ("A 15% drop under perturbation.", {"clean_score": 0.9, "perturbed_score": 0.75}),
        ("A severe 50% drop (clipped).", {"clean_score": 0.9, "perturbed_score": 0.4}),
        ("No perturbed_score on the record.", {"clean_score": 0.9}),
        (
            "Case was cancelled (for example an audit stop).",
            {"clean_score": 0.9, "perturbed_score": 0.9, "cancelled": True},
        ),
    )
    for story, record in examples:
        result = score_modality_robustness(record)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_modality_robustness_examples()
