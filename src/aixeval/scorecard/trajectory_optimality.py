"""trajectory_optimality raw score: a judge verdict on visible path waste.

Known limitation, acknowledged not hidden: a judge reading a trace has no
access to the true optimal path for a task, so it cannot certify a
trajectory as genuinely shortest; it can only flag *visible* waste
(a step obviously repeated with no new information, work redone, or a
step that clearly did not contribute to the output). The prompt in
``aixeval.judge.trajectory_optimality`` is written to ask only for
that, not for a claim of true optimality. Treat a high score as "nothing
obviously wasteful," not "provably the best possible path."
"""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "trajectory_optimality"
# A judge (human or model) rates visible path waste on [0, 1]. This module
# does not call a model. It grades a stored or injected verdict.
TrajectoryOptimalityJudge = Callable[[str, str], float]


def score_trajectory_optimality(
    record: dict[str, Any],
    judge: TrajectoryOptimalityJudge | None = None,
) -> dict[str, Any]:
    """Score trajectory optimality from a judge verdict on ``[0, 1]``.

    Prefers ``trajectory_optimality_judge`` already on the record. If
    that field is missing and ``judge`` is passed, calls
    ``judge(trajectory, output)``. The toolkit does not ship a model
    client. Missing or invalid verdicts are N/A, never a silent 0.0.
    """
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("cancelled")

    if "trajectory_optimality_judge" in record and record["trajectory_optimality_judge"] is not None:
        return _from_verdict(record["trajectory_optimality_judge"])

    trajectory, trajectory_ok = _optional_text(record, "trajectory")
    output, output_ok = _optional_text(record, "output")
    if judge is None or not trajectory_ok or not output_ok or trajectory is None:
        return _na("missing_evidence")
    if output is None:
        return _na("missing_output")
    return _from_verdict(judge(trajectory, output))


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


def print_trajectory_optimality_examples() -> None:
    """Print how a judge verdict becomes a raw score and grade."""
    print("trajectory_optimality asks: did the steps taken visibly waste effort reaching the output?")
    print("  A judge (model or human) returns trajectory_optimality_judge on [0, 1].")
    print("  This function only grades that verdict. It does not call a model.")
    print("  A scored row also gets a 0-4 grade from the declared higher-is-better table.")
    print("  No verdict, or a bad number, is na. No grade on N/A.")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("Judge says no visible waste.", {"trajectory_optimality_judge": 1.0}),
        ("Judge says clearly wasteful (repeated steps).", {"trajectory_optimality_judge": 0.0}),
        ("Judge says some waste (raw 0.6 -> grade 2).", {"trajectory_optimality_judge": 0.6}),
        ("No judge verdict on the record.", {"output": "Paris", "trajectory": "1. search -- Paris"}),
        (
            "Case was cancelled (for example an audit stop).",
            {"trajectory_optimality_judge": 1.0, "cancelled": True},
        ),
    )
    for story, record in examples:
        result = score_trajectory_optimality(record)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_trajectory_optimality_examples()
