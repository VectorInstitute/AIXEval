"""delegation_accuracy raw score: share of gold subtasks sent to the correct agent.

A multi-agent-system construct, declared against a gold role
assignment: the overlay supplies the gold roles and the trace
supplies handoffs. A handoff counts as correct when its text equals the
subtask and its recipient equals the gold agent. Without handoff text,
the overlay field is not built and this metric is N/A.
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


def score_delegation_accuracy(record: dict[str, Any]) -> dict[str, Any]:
    """Score the share of gold subtasks handled by the correct agent."""
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("delegation_accuracy", "cancelled")
    correct, correct_ok = _optional_count(record, "subtasks_correct")
    total, total_ok = _optional_count(record, "subtasks_total")
    if not correct_ok or not total_ok or correct is None or total is None:
        return _na("delegation_accuracy", "missing_evidence")
    if total == 0:
        return _na("delegation_accuracy", "no_gold")
    return _scored("delegation_accuracy", min(1.0, correct / total), higher_is_better=True)
