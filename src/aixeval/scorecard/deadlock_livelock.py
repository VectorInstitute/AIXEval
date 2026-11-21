"""deadlock_livelock raw score: did the run stall or loop.

A multi-agent-system construct: did the run stall or loop. The overlay
declares ``run_stalled``; the trace's repeated-call evidence is advisory
only, since the threshold is a declared choice. Lower is better.
"""

from __future__ import annotations

from typing import Any

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade


def _optional_count(record: dict[str, Any], key: str) -> tuple[int | None, bool]:
    if key not in record or record[key] is None:
        return None, True
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None, False
    return value, True


def _scored(metric_id: str, raw_score: float, higher_is_better: bool) -> dict[str, Any]:
    return apply_grade(
        {"metric_id": metric_id, "status": "scored", "raw_score": raw_score},
        higher_is_better=higher_is_better,
    )


def _na(metric_id: str, reason: str) -> dict[str, Any]:
    return {"metric_id": metric_id, "status": "na", "na_reason": require_na_reason(reason)}


def score_deadlock_livelock(record: dict[str, Any]) -> dict[str, Any]:
    """Score whether the run stalled, from the declared flag."""
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("deadlock_livelock", "cancelled")
    stalled = record.get("run_stalled")
    if not isinstance(stalled, bool):
        return _na("deadlock_livelock", "missing_evidence")
    return _scored("deadlock_livelock", 1.0 if stalled else 0.0, higher_is_better=False)
