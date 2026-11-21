"""verbalized_confidence raw score: the stated [0, 1], then a 0-4 grade."""

from __future__ import annotations

from math import isfinite
from typing import Any

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "verbalized_confidence"
# First slice: the number the model stated. Not calibration vs gold.


def score_verbalized_confidence(record: dict[str, Any]) -> dict[str, Any]:
    """Score the confidence the model stated, when it stated one.

    Reads ``stated_confidence`` as a finite number on ``[0, 1]``.
    That value is the raw score (higher-is-better). This is not
    calibration against the gold answer. Missing or invalid values
    are N/A, never a silent 0.0.
    """
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("cancelled")

    if "stated_confidence" not in record or record["stated_confidence"] is None:
        return _na("missing_evidence")
    return _from_stated(record["stated_confidence"])


def _from_stated(value: object) -> dict[str, Any]:
    """Accept a finite number on ``[0, 1]``. Anything else is N/A."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return _na("missing_evidence")
    raw_score = float(value)
    if not isfinite(raw_score) or raw_score < 0.0 or raw_score > 1.0:
        return _na("missing_evidence")
    return apply_grade({"metric_id": METRIC_ID, "status": "scored", "raw_score": raw_score})


def _na(reason: str) -> dict[str, Any]:
    return {"metric_id": METRIC_ID, "status": "na", "na_reason": require_na_reason(reason)}


def print_verbalized_confidence_examples() -> None:
    """Print how a stated confidence becomes a raw score and grade."""
    print("verbalized_confidence asks: what confidence did the model state, when it stated one?")
    print("  stated_confidence on [0, 1] is the raw_score. Higher-is-better.")
    print("  This is the stated number, not calibration against gold.")
    print("  No statement or a bad number -> na, not a silent 0.")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("Model said it was sure (1.0).", {"stated_confidence": 1.0}),
        ("Model said 0.7.", {"stated_confidence": 0.7}),
        ("Model said it was unsure (0.0).", {"stated_confidence": 0.0}),
        ("No stated confidence.", {"output": "yes"}),
        (
            "Case was cancelled (for example an audit stop).",
            {"stated_confidence": 1.0, "cancelled": True},
        ),
    )
    for story, record in examples:
        result = score_verbalized_confidence(record)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_verbalized_confidence_examples()
