"""error_recovery raw score: recovered after a failed step, then a 0-4 grade."""

from __future__ import annotations

from typing import Any

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "error_recovery"
# Declared first slice: a failed step, then whether the run recovered.
# Ingest must set the flags. This function does not walk a trace.


def score_error_recovery(record: dict[str, Any]) -> dict[str, Any]:
    """Score recovery after a failed step or dead end.

    Reads ``had_failure`` and, when that is true, ``recovered``. Both must
    be bools. No failed step is N/A ``not_in_scope``. Missing flags are
    N/A, never a silent 0.0. Recovered is 1.0; failed and stuck is 0.0.
    """
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("cancelled")

    had_failure, had_ok = _optional_bool(record, "had_failure")
    if not had_ok:
        return _na("missing_evidence")
    if had_failure is None:
        return _na("missing_evidence")
    if had_failure is False:
        return _na("not_in_scope")

    recovered, recovered_ok = _optional_bool(record, "recovered")
    if not recovered_ok or recovered is None:
        return _na("missing_evidence")
    return _scored(1.0 if recovered else 0.0)


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


def print_error_recovery_examples() -> None:
    """Print how failed-step flags become a raw score and grade."""
    print("error_recovery asks: after a failed step or dead end, did the run recover?")
    print("  had_failure = there was a failed tool, error span, or dead end")
    print("  recovered   = the run continued usefully after that (not stuck on the error)")
    print("  Failure then recover -> raw_score 1.0. Failure and stuck -> 0.0.")
    print("  No failed step -> na not_in_scope. Missing flags -> na, not a silent 0.")
    print("  A scored row also gets a 0-4 grade from the declared higher-is-better table.")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("Tool failed, then the run recovered.", {"had_failure": True, "recovered": True}),
        ("Tool failed, and the run died on that error.", {"had_failure": True, "recovered": False}),
        ("Clean run, no failed step.", {"had_failure": False}),
        ("No flags. Cannot score.", {"output": "yes"}),
        ("Failed step, but no recovered flag.", {"had_failure": True}),
        (
            "Case was cancelled (for example an audit stop).",
            {"had_failure": True, "recovered": True, "cancelled": True},
        ),
    )
    for story, record in examples:
        result = score_error_recovery(record)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_error_recovery_examples()
