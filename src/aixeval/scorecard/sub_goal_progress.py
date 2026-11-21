"""sub_goal_progress raw score: fraction of declared sub-goals completed.

Literature: Ma, Zhang, Zhu, Yang, Yang, Jin, Lan, Kong, and He,
*AgentBoard: An Analytical Evaluation Board of Multi-turn LLM Agents*,
NeurIPS 2024 (arXiv:2401.13178, verified via search, not guessed),
introduces a fine-grained progress-rate / subgoal-completion-rate
metric tracking intermediate agent progress, rather than a single
all-or-nothing success flag.

Declared, not computed: AgentBoard's own mechanism determines subgoal
completion with an LLM-based postcondition checker reading state and
action history. This toolkit does not run that checker; it reads a
declared ``subgoals_completed``/``subgoals_total`` count, the same
discipline as ``cost_per_query``/``cross_modal_retrieval``, a fact
some other process already determined.
"""

from __future__ import annotations

from typing import Any

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "sub_goal_progress"


def score_sub_goal_progress(record: dict[str, Any]) -> dict[str, Any]:
    """Score the fraction of declared sub-goals completed.

    Reads ``subgoals_completed`` (int >= 0) and ``subgoals_total``
    (int >= 1). Raw score is ``subgoals_completed / subgoals_total``,
    clipped to ``[0, 1]`` (a completed count above the declared total
    is clamped, not rejected). Higher is better. Missing or invalid
    counts are N/A, never a silent 0.0.
    """
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("cancelled")

    completed, completed_ok = _nonneg_int(record, "subgoals_completed")
    total, total_ok = _positive_int(record, "subgoals_total")
    if not completed_ok or not total_ok or completed is None or total is None:
        return _na("missing_evidence")
    raw_score = min(1.0, max(0.0, completed / total))
    return _scored(raw_score)


def _nonneg_int(record: dict[str, Any], key: str) -> tuple[int | None, bool]:
    """Return an int >= 0 and whether the field type is usable."""
    if key not in record or record[key] is None:
        return None, True
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, int):
        return None, False
    if value < 0:
        return None, False
    return value, True


def _positive_int(record: dict[str, Any], key: str) -> tuple[int | None, bool]:
    """Return an int >= 1 and whether the field type is usable."""
    if key not in record or record[key] is None:
        return None, True
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, int):
        return None, False
    if value < 1:
        return None, False
    return value, True


def _scored(raw_score: float) -> dict[str, Any]:
    return apply_grade({"metric_id": METRIC_ID, "status": "scored", "raw_score": raw_score})


def _na(reason: str) -> dict[str, Any]:
    return {"metric_id": METRIC_ID, "status": "na", "na_reason": require_na_reason(reason)}


def print_sub_goal_progress_examples() -> None:
    """Print how a declared sub-goal count becomes a raw score and grade."""
    print("sub_goal_progress asks: what fraction of the declared sub-goals got completed?")
    print("  raw_score = subgoals_completed / subgoals_total, clipped to [0, 1]")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("All 4 sub-goals completed.", {"subgoals_completed": 4, "subgoals_total": 4}),
        ("3 of 4 sub-goals completed.", {"subgoals_completed": 3, "subgoals_total": 4}),
        ("None of 4 sub-goals completed.", {"subgoals_completed": 0, "subgoals_total": 4}),
        ("No sub-goal count on the record.", {"output": "yes"}),
        ("Zero declared total: cannot score.", {"subgoals_completed": 0, "subgoals_total": 0}),
        (
            "Case was cancelled (for example an audit stop).",
            {"subgoals_completed": 4, "subgoals_total": 4, "cancelled": True},
        ),
    )
    for story, record in examples:
        result = score_sub_goal_progress(record)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_sub_goal_progress_examples()
