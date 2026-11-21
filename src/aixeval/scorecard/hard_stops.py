"""Deterministic account of the hard stops that fired on one card.

No model call: everything here is read straight off the scored rows and
the sealed record, so it is as reproducible as the card itself.
"""

from __future__ import annotations

from typing import Any

from aixeval.scorecard.definitions import HARD_STOP_DEFINITIONS
from aixeval.scorecard.profile import DIMENSIONS, HARD_STOP_RULES, HardStopRule, build_profile


# Record fields each hard-stop metric reads; the facts that made it fire.
HARD_STOP_INPUT_FIELDS: dict[str, tuple[str, ...]] = {
    "contamination": ("contaminated",),
    "unsafe_action": ("high_impact_in_scope", "unsafe_taken"),
    "confirmation_gating": ("high_impact_in_scope", "confirmed"),
}

_EFFECT_TEXT: dict[str, str] = {
    "void_capability": "capability withdrawn",
    "cap_0": "every filled question capped at 0",
    "cap_1": "every filled question capped at 1",
}


def explain_hard_stops(
    rows: list[dict[str, Any]],
    record: dict[str, Any],
    *,
    hard_stop_rules: dict[str, HardStopRule] = HARD_STOP_RULES,
) -> dict[str, Any]:
    """Explain which hard stops fired, on what evidence, and what they capped.

    Parameters
    ----------
    rows
        Graded metric rows, the same list ``build_profile`` takes.
    record
        The sealed record those rows were scored from.
    hard_stop_rules
        The policy to explain; defaults to the shipped one.

    Returns
    -------
    dict
        ``hard_stops``: one entry per fired rule (name, definition,
        metric, raw score, rule text, effect, the record fields it read).
        ``capped_questions``: ``{question: {"before": grade, "after": grade}}``
        for every question the stops changed; ``after`` is ``None`` when
        capability was withdrawn. Both are empty when nothing fired.
    """
    profile = build_profile(rows, hard_stop_rules=hard_stop_rules)
    uncapped = build_profile(rows, hard_stop_rules=_never_firing(hard_stop_rules))
    stops = []
    for name in profile["applied_hard_stops"]:
        rule = hard_stop_rules[name]
        metric_id = rule["metric_id"]
        comparator = ">=" if rule["compare"] == "ge" else "<"
        fields = HARD_STOP_INPUT_FIELDS.get(metric_id, ())
        stops.append(
            {
                "hard_stop": name,
                "definition": HARD_STOP_DEFINITIONS.get(name, ""),
                "metric_id": metric_id,
                "raw_score": _raw_score(rows, metric_id),
                "rule": f"{metric_id} raw score {comparator} {rule['cutoff']:.2f}",
                "effect": _EFFECT_TEXT[rule["effect"]],
                "evidence_fields": {field: record[field] for field in fields if field in record},
            }
        )
    capped = {
        question: {"before": uncapped[question], "after": profile[question]}
        for question in DIMENSIONS
        if uncapped[question] != profile[question]
    }
    return {"hard_stops": stops, "capped_questions": capped}


def _never_firing(rules: dict[str, HardStopRule]) -> dict[str, HardStopRule]:
    """Return the same rules with cutoffs no raw score in [0, 1] can trip."""
    return {
        name: HardStopRule(
            metric_id=rule["metric_id"],
            compare=rule["compare"],
            cutoff=2.0 if rule["compare"] == "ge" else -1.0,
            effect=rule["effect"],
        )
        for name, rule in rules.items()
    }


def _raw_score(rows: list[dict[str, Any]], metric_id: str) -> float | None:
    for row in rows:
        if row.get("metric_id") == metric_id and row.get("status") == "scored":
            return float(row["raw_score"])
    return None
