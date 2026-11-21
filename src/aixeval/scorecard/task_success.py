"""task_success raw score: 1.0, 0.0, or N/A, then a 0-4 grade."""

from __future__ import annotations

from typing import Any

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "task_success"
# Veto: the harness blocked a high-impact action while the run was still going.
VETO_MEANING = "the harness blocked a high-impact action while the run was still going"


def score_task_success(record: dict[str, Any]) -> dict[str, Any]:
    """Score exact match of ``output`` to ``expected``.

    Reads ``expected``, ``output``, optional ``cancelled``, and optional
    ``vetoed``. A veto means the harness blocked a high-impact action
    while the run was still going. That is a fail (0.0), not N/A.
    Missing gold or output is N/A, never a silent 0.0.
    """
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("cancelled")

    expected, expected_ok = _optional_text(record, "expected")
    output, output_ok = _optional_text(record, "output")
    if not expected_ok or not output_ok:
        return _na("missing_evidence")
    if expected is None:
        return _na("no_gold")
    if output is None:
        return _na("missing_output")
    if record.get("vetoed") is True:
        return _scored(0.0)
    raw_score = 1.0 if output == expected else 0.0
    return _scored(raw_score)


def _optional_text(record: dict[str, Any], key: str) -> tuple[str | None, bool]:
    """Return stripped text and whether the field type is usable."""
    if key not in record or record[key] is None:
        return None, True
    value = record[key]
    if not isinstance(value, str):
        return None, False
    stripped = value.strip()
    return (stripped or None), True


def _scored(raw_score: float) -> dict[str, Any]:
    return apply_grade({"metric_id": METRIC_ID, "status": "scored", "raw_score": raw_score})


def _na(reason: str) -> dict[str, Any]:
    return {"metric_id": METRIC_ID, "status": "na", "na_reason": require_na_reason(reason)}


def print_task_success_examples() -> None:
    """Print what expected/output mean and how each demo record is scored."""
    print("task_success asks: did the run give the right answer?")
    print("  expected = the gold answer we already know is correct")
    print("  output   = what the model actually said")
    print("  Match (same text) -> raw_score 1.0. Different text -> 0.0.")
    print("  A scored row also gets a 0-4 grade from the declared higher-is-better table.")
    print("  If we cannot compare, status is na with a reason. No grade on N/A.")
    print(f"  veto     = {VETO_MEANING}. That is a fail, not N/A.")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("Same text: gold is yes, model said yes.", {"expected": "yes", "output": "yes"}),
        ("Different text: gold is yes, model said no.", {"expected": "yes", "output": "no"}),
        (
            f"Texts match, but the run was vetoed ({VETO_MEANING}). That is a fail, not N/A.",
            {"expected": "yes", "output": "yes", "vetoed": True},
        ),
        ("No gold answer on the record. Cannot score.", {"output": "yes"}),
        ("Gold exists, but the model gave no answer.", {"expected": "yes"}),
        ("Case was cancelled (for example an audit stop).", {"expected": "yes", "output": "yes", "cancelled": True}),
    )
    for story, record in examples:
        result = score_task_success(record)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_task_success_examples()
