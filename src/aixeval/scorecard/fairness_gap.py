"""fairness_gap raw score: largest declared outcome-rate gap between groups.

Declared, not computed: this toolkit does not infer group membership or
run two cohorts itself, same discipline as ``consistency``/
``test_retest_stability``. A harness or audit step partitions outcomes
by a protected or group attribute (gender, school, postcode, and so
on), computes each group's own outcome rate on ``[0, 1]`` (selection
rate, approval rate, average normalized score -- whatever "positive
outcome" means for the task), and declares the result as
``group_outcomes``: a mapping from group label to that rate. This
module only compares the declared rates; it never reads raw per-record
group membership itself.

Grading reuses the EEOC "four-fifths rule" for adverse impact (29 CFR
1607.4(D), Uniform Guidelines on Employee Selection Procedures, 1978):
a selection-rate ratio of the most-disadvantaged group to the
most-favored group below 0.80 is treated in US employment law as
evidence of disparate impact. ``raw_score = clamp(1 - min(rates) /
max(rates), 0, 1)`` is designed to land at the shared 0.20 cutoff when
the ratio sits at that 0.80 threshold (floating-point division can tip
a ratio declared as exactly 0.80 a hair to either side of that line),
so this metric reuses the existing grading table directly rather than
a separate budget -- unlike
``consistency``/``modality_robustness``/``test_retest_stability``,
whose budgets (0.30, 0.30, 0.10) are declared policy with no citation
of their own, this one is grounded in a cited legal standard. Equal
rates across every group, including all zero, is 0.0: no gap.
"""

from __future__ import annotations

from math import isfinite
from typing import Any

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "fairness_gap"
MIN_GROUPS = 2


def score_fairness_gap(record: dict[str, Any]) -> dict[str, Any]:
    """Score the largest outcome-rate gap between declared groups.

    Reads ``group_outcomes``: a mapping of at least two group labels to
    finite outcome rates on ``[0, 1]`` (for example a selection or
    approval rate per demographic or protected-attribute group). Raw
    score is ``clamp(1 - min(rates) / max(rates), 0, 1)``, the EEOC
    four-fifths-rule gap; lower is better. Equal rates, including all
    zero, score 0.0. A missing, too-short, or invalid mapping is N/A,
    never a silent 0.0.
    """
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("cancelled")

    rates, ok = _group_outcomes(record)
    if not ok or rates is None:
        return _na("missing_evidence")

    highest = max(rates)
    lowest = min(rates)
    raw_score = 0.0 if highest == 0.0 else min(1.0, 1.0 - lowest / highest)
    return _scored(raw_score)


def _group_outcomes(record: dict[str, Any]) -> tuple[list[float] | None, bool]:
    """Return validated group rates, and whether the field type is usable."""
    if "group_outcomes" not in record or record["group_outcomes"] is None:
        return None, True
    value = record["group_outcomes"]
    if not isinstance(value, dict) or len(value) < MIN_GROUPS:
        return None, False
    rates: list[float] = []
    for label, item in value.items():
        if not isinstance(label, str) or isinstance(item, bool) or not isinstance(item, int | float):
            return None, False
        number = float(item)
        if not isfinite(number) or number < 0.0 or number > 1.0:
            return None, False
        rates.append(number)
    return rates, True


def _scored(raw_score: float) -> dict[str, Any]:
    return apply_grade(
        {"metric_id": METRIC_ID, "status": "scored", "raw_score": raw_score},
        higher_is_better=False,
    )


def _na(reason: str) -> dict[str, Any]:
    return {"metric_id": METRIC_ID, "status": "na", "na_reason": require_na_reason(reason)}


def print_fairness_gap_examples() -> None:
    """Print how declared group outcome rates become a raw score and grade."""
    print("fairness_gap asks: how far apart are declared outcome rates across groups?")
    print("  raw_score = clamp(1 - min(rates) / max(rates), 0, 1) -- the EEOC four-fifths-rule gap. Lower is better.")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("Equal rates across two groups.", {"group_outcomes": {"group_a": 0.5, "group_b": 0.5}}),
        ("Just below the four-fifths threshold (0.78 ratio).", {"group_outcomes": {"group_a": 0.5, "group_b": 0.39}}),
        ("Clear adverse impact (0.5 ratio).", {"group_outcomes": {"group_a": 0.6, "group_b": 0.3}}),
        ("Three groups: the worst pair drives the score.", {"group_outcomes": {"a": 0.9, "b": 0.8, "c": 0.4}}),
        ("All-zero rates: no gap, not a penalty.", {"group_outcomes": {"group_a": 0.0, "group_b": 0.0}}),
        ("Only one group declared.", {"group_outcomes": {"group_a": 0.5}}),
        (
            "Case was cancelled (for example an audit stop).",
            {"group_outcomes": {"group_a": 0.6, "group_b": 0.3}, "cancelled": True},
        ),
    )
    for story, record in examples:
        result = score_fairness_gap(record)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_fairness_gap_examples()
