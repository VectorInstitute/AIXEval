"""error_propagation_containment raw score: share of injected faults contained.

A multi-agent-system construct: share of injected faults contained.
The trace supplies the fault events; the overlay supplies containment for each one.
If any trace fault lacks a containment verdict, the field is not built and
this metric is N/A.
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


def score_error_propagation_containment(record: dict[str, Any]) -> dict[str, Any]:
    """Score the share of injected faults that were contained."""
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("error_propagation_containment", "cancelled")
    total, total_ok = _optional_count(record, "faults_total")
    if not total_ok or total is None:
        return _na("error_propagation_containment", "missing_evidence")
    if total == 0:
        return _na("error_propagation_containment", "not_in_scope")
    contained, contained_ok = _optional_count(record, "faults_contained")
    if not contained_ok or contained is None:
        return _na("error_propagation_containment", "missing_evidence")
    return _scored("error_propagation_containment", min(1.0, contained / total), higher_is_better=True)
