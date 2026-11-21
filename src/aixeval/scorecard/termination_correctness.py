"""termination_correctness raw score: did the run stop at the correct condition.

Declared, not computed: the harness that ran the agents knows its own
terminal condition (a goal reached, a budget exhausted, a stop token).
It declares whether the run ended at the correct one
(``terminated_correctly``). This module only grades that fact.

A multi-agent-system construct, built here on request ahead of this
toolkit's own usual scoping for MAS metrics.
"""

from __future__ import annotations

from typing import Any

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "termination_correctness"


def score_termination_correctness(record: dict[str, Any]) -> dict[str, Any]:
    """Score whether the run terminated at the correct condition.

    Reads ``terminated_correctly`` (bool). True is 1.0, False is 0.0.
    Higher is better. A missing or non-bool flag is N/A, never a silent 0.0.
    """
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("cancelled")

    flag, flag_ok = _optional_bool(record, "terminated_correctly")
    if not flag_ok or flag is None:
        return _na("missing_evidence")
    return _scored(1.0 if flag else 0.0)


def _optional_bool(record: dict[str, Any], key: str) -> tuple[bool | None, bool]:
    """Return a bool and whether the field type is usable."""
    if key not in record or record[key] is None:
        return None, True
    value = record[key]
    if not isinstance(value, bool):
        return None, False
    return value, True


def _scored(raw_score: float) -> dict[str, Any]:
    return apply_grade({"metric_id": METRIC_ID, "status": "scored", "raw_score": raw_score})


def _na(reason: str) -> dict[str, Any]:
    return {"metric_id": METRIC_ID, "status": "na", "na_reason": require_na_reason(reason)}


def print_termination_correctness_examples() -> None:
    """Print how the terminal-condition flag becomes a raw score and grade."""
    print("termination_correctness asks: did the run stop at the correct terminal condition?")
    print("  terminated_correctly = the harness confirms the run ended at its correct condition")
    print("  True -> raw_score 1.0 (grade 4). False -> raw_score 0.0 (grade 0).")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("Stopped at the goal condition.", {"terminated_correctly": True}),
        ("Stopped early, or looped past the stop token.", {"terminated_correctly": False}),
        ("No flag on the record.", {"output": "yes"}),
        (
            "Case was cancelled (for example an audit stop).",
            {"terminated_correctly": True, "cancelled": True},
        ),
    )
    for story, record in examples:
        result = score_termination_correctness(record)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_termination_correctness_examples()
