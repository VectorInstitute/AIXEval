"""contamination raw score: leak or clean, then a lower-is-better 0-4 grade."""

from __future__ import annotations

from typing import Any

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "contamination"
# Declared first slice: a bool on the sealed record. Ingest must set it.
# This function does not scan the prompt or call a judge.


def score_contamination(record: dict[str, Any]) -> dict[str, Any]:
    """Score whether gold or other test material leaked into the run.

    Reads ``contaminated`` as a bool. True is raw score 1.0 (leak).
    False is 0.0 (clean). Grade uses the lower-is-better table: clean
    is Excellent. Missing or non-bool is N/A, never a silent 0.0.
    """
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("cancelled")

    contaminated, ok = _optional_bool(record, "contaminated")
    if not ok or contaminated is None:
        return _na("missing_evidence")
    return _scored(1.0 if contaminated else 0.0)


def _optional_bool(record: dict[str, Any], key: str) -> tuple[bool | None, bool]:
    """Return a bool and whether the field type is usable."""
    if key not in record or record[key] is None:
        return None, True
    value = record[key]
    if not isinstance(value, bool):
        return None, False
    return value, True


def _scored(raw_score: float) -> dict[str, Any]:
    return apply_grade(
        {"metric_id": METRIC_ID, "status": "scored", "raw_score": raw_score},
        higher_is_better=False,
    )


def _na(reason: str) -> dict[str, Any]:
    return {"metric_id": METRIC_ID, "status": "na", "na_reason": require_na_reason(reason)}


def print_contamination_examples() -> None:
    """Print how the contaminated flag becomes a raw score and grade."""
    print("contamination asks: was gold or other test material already in the prompt or overlap?")
    print("  contaminated True  -> raw_score 1.0 (leak). Lower-is-better grade 0.")
    print("  contaminated False -> raw_score 0.0 (clean). Lower-is-better grade 4.")
    print("  Missing or non-bool flag -> na, not a silent 0.")
    print("  If raw_score is at least 0.40, build_profile voids capability (declared policy).")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("Clean: gold was not in the prompt.", {"contaminated": False}),
        ("Leak: gold or test overlap was already there.", {"contaminated": True}),
        ("No flag. Cannot score.", {"expected": "yes", "output": "yes"}),
        (
            "Case was cancelled (for example an audit stop).",
            {"contaminated": False, "cancelled": True},
        ),
    )
    for story, record in examples:
        result = score_contamination(record)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_contamination_examples()
