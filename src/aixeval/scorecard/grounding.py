"""grounding raw score: a faithfulness judge verdict, then a 0-4 grade."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "grounding"
# A judge (human or model) rates faithfulness of output to evidence on [0, 1].
# This module does not call a model. It grades a stored or injected verdict.
GroundingJudge = Callable[[str, str], float]


def score_grounding(
    record: dict[str, Any],
    judge: GroundingJudge | None = None,
) -> dict[str, Any]:
    """Score grounding from a judge verdict on ``[0, 1]``.

    Prefers ``grounding_judge`` already on the record. If that field is
    missing and ``judge`` is passed, calls ``judge(evidence, output)``.
    The toolkit does not ship a model client. Missing or invalid verdicts
    are N/A, never a silent 0.0.
    """
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("cancelled")

    if "grounding_judge" in record and record["grounding_judge"] is not None:
        return _from_verdict(record["grounding_judge"])

    evidence, evidence_ok = _optional_text(record, "evidence")
    output, output_ok = _optional_text(record, "output")
    if judge is None or not evidence_ok or not output_ok or evidence is None:
        return _na("missing_evidence")
    if output is None:
        return _na("missing_output")
    return _from_verdict(judge(evidence, output))


def _from_verdict(value: object) -> dict[str, Any]:
    """Accept a finite number on ``[0, 1]``. Anything else is N/A."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return _na("missing_evidence")
    raw_score = float(value)
    if not isfinite(raw_score) or raw_score < 0.0 or raw_score > 1.0:
        return _na("missing_evidence")
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


def print_grounding_examples() -> None:
    """Print how a judge verdict becomes a raw score and grade."""
    print("grounding asks: did the answer stay faithful to the given evidence?")
    print("  A judge (model or human) returns grounding_judge on [0, 1].")
    print("  This function only grades that verdict. It does not call a model.")
    print("  A scored row also gets a 0-4 grade from the declared higher-is-better table.")
    print("  No verdict, or a bad number, is na. No grade on N/A.")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("Judge says fully supported.", {"grounding_judge": 1.0}),
        ("Judge says unsupported.", {"grounding_judge": 0.0}),
        ("Judge says mostly supported (raw 0.7 -> grade 3).", {"grounding_judge": 0.7}),
        ("No judge verdict on the record.", {"output": "yes", "evidence": "yes"}),
        ("Case was cancelled (for example an audit stop).", {"grounding_judge": 1.0, "cancelled": True}),
    )
    for story, record in examples:
        result = score_grounding(record)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_grounding_examples()
