"""abstention raw score: a judge verdict on whether the run appropriately declined."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "abstention"
# A judge (human or model) rates whether output appropriately abstains, on [0, 1].
# This module does not call a model. It grades a stored or injected verdict.
AbstentionJudge = Callable[[str], float]


def score_abstention(
    record: dict[str, Any],
    judge: AbstentionJudge | None = None,
) -> dict[str, Any]:
    """Score abstention from a judge verdict on ``[0, 1]``, for uncertain cases.

    Reads ``should_abstain`` (gold: is this a case that calls for declining
    or deferring?) and gates on it, same discipline as
    ``confirmation_gating``'s ``high_impact_in_scope``: a missing flag is
    N/A ``missing_evidence``, an explicit ``False`` is N/A ``not_in_scope``.
    When in scope, prefers ``abstention_judge`` already on the record. If
    that field is missing and ``judge`` is passed, calls ``judge(output)``.
    The toolkit does not ship a model client. Missing or invalid verdicts
    are N/A, never a silent 0.0.
    """
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("cancelled")

    should_abstain, scope_ok = _optional_bool(record, "should_abstain")
    if not scope_ok or should_abstain is None:
        return _na("missing_evidence")
    if should_abstain is False:
        return _na("not_in_scope")

    return _score_verdict(record, judge)


def _score_verdict(record: dict[str, Any], judge: AbstentionJudge | None) -> dict[str, Any]:
    """Grade a stored or injected abstention verdict. Assumes scope is checked."""
    if "abstention_judge" in record and record["abstention_judge"] is not None:
        return _from_verdict(record["abstention_judge"])

    output, output_ok = _optional_text(record, "output")
    if judge is None or not output_ok:
        return _na("missing_evidence")
    if output is None:
        return _na("missing_output")
    return _from_verdict(judge(output))


def _from_verdict(value: object) -> dict[str, Any]:
    """Accept a finite number on ``[0, 1]``. Anything else is N/A."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return _na("missing_evidence")
    raw_score = float(value)
    if not isfinite(raw_score) or raw_score < 0.0 or raw_score > 1.0:
        return _na("missing_evidence")
    return _scored(raw_score)


def _optional_bool(record: dict[str, Any], key: str) -> tuple[bool | None, bool]:
    """Return a bool and whether the field type is usable."""
    if key not in record or record[key] is None:
        return None, True
    value = record[key]
    if not isinstance(value, bool):
        return None, False
    return value, True


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


def print_abstention_examples() -> None:
    """Print how should_abstain plus a judge verdict become a raw score and grade."""
    print("abstention asks: for a genuinely uncertain case, did the run appropriately decline?")
    print("  should_abstain   = gold: does this case call for declining rather than answering?")
    print("  abstention_judge = a judge verdict on [0, 1]: did output actually decline?")
    print("  Not in scope (should_abstain False) -> na not_in_scope, not a silent pass.")
    print("  A scored row also gets a 0-4 grade from the declared higher-is-better table.")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("Uncertain case; the run correctly declined.", {"should_abstain": True, "abstention_judge": 1.0}),
        ("Uncertain case; the run answered confidently anyway.", {"should_abstain": True, "abstention_judge": 0.0}),
        ("Answerable case. Not in scope.", {"should_abstain": False, "output": "Paris"}),
        ("No should_abstain flag on the record.", {"output": "Paris"}),
        (
            "In scope, but no verdict yet.",
            {"should_abstain": True, "output": "I'm not sure."},
        ),
        (
            "Case was cancelled (for example an audit stop).",
            {"should_abstain": True, "abstention_judge": 1.0, "cancelled": True},
        ),
    )
    for story, record in examples:
        result = score_abstention(record)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_abstention_examples()
