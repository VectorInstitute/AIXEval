"""Declared overlay for a multi-agent run: ground truth no trace contains.

An overlay is a separate annotation, written by the harness or a person,
keyed to one run. ``build_overlay_fields`` combines it with the spans
derived from the trace and returns flat record fields for the metrics.

A field is left out when its evidence is incomplete, so the metric reports
N/A instead of scoring a guess.
"""

from __future__ import annotations

from typing import Any

from aixeval.ingest.trace import Span, derive_delegations, derive_faults


def validate_overlay(overlay: dict[str, Any]) -> None:
    """Check the overlay's shape and types. Unknown top-level keys are rejected."""
    allowed = {"faults", "gold_roles", "run_stalled", "mas_unsafe_steps", "mas_total_steps"}
    unknown = sorted(set(overlay) - allowed)
    if unknown:
        msg = f"unknown overlay key(s): {', '.join(unknown)}"
        raise ValueError(msg)
    for item in _list(overlay, "faults"):
        if not isinstance(item.get("span_id"), str) or not isinstance(item.get("contained"), bool):
            msg = "each fault needs a string span_id and a bool contained"
            raise ValueError(msg)
    for item in _list(overlay, "gold_roles"):
        if not isinstance(item.get("subtask"), str) or not isinstance(item.get("agent"), str):
            msg = "each gold role needs a string subtask and a string agent"
            raise ValueError(msg)
    if "run_stalled" in overlay and not isinstance(overlay["run_stalled"], bool):
        msg = "run_stalled must be a bool"
        raise ValueError(msg)
    for key in ("mas_unsafe_steps", "mas_total_steps"):
        value = overlay.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            msg = f"{key} must be an int >= 0"
            raise ValueError(msg)


def build_overlay_fields(spans: list[Span], overlay: dict[str, Any]) -> dict[str, Any]:
    """Return flat record fields from the trace and overlay.

    Incomplete evidence is omitted, so the metric reports N/A.
    """
    validate_overlay(overlay)
    fields: dict[str, Any] = {}

    faults = derive_faults(spans)
    fields["faults_total"] = len(faults)
    contained_by_span = {item["span_id"]: item["contained"] for item in _list(overlay, "faults")}
    if all(f["span_id"] in contained_by_span for f in faults):
        fields["faults_contained"] = sum(contained_by_span[f["span_id"]] for f in faults)

    gold_roles = _list(overlay, "gold_roles")
    fields["subtasks_total"] = len(gold_roles)
    delegations = derive_delegations(spans)
    if gold_roles and all(d["text"] is not None for d in delegations):
        correct = sum(
            1
            for role in gold_roles
            if any(d["text"] == role["subtask"] and d["recipient"] == role["agent"] for d in delegations)
        )
        fields["subtasks_correct"] = correct

    if "run_stalled" in overlay:
        fields["run_stalled"] = overlay["run_stalled"]
    for key in ("mas_unsafe_steps", "mas_total_steps"):
        if overlay.get(key) is not None:
            fields[key] = overlay[key]
    return fields


def _list(overlay: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = overlay.get(key)
    return value if isinstance(value, list) else []
