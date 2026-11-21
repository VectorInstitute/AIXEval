"""tool_call_accuracy raw score: a verdict on whether the right tools+args were used.

The verdict can come from either of two independent, interchangeable
mechanisms: this scorer does not know or care which one produced it,
the same swappability already established for ``should_abstain`` /
``should_refuse``:

- **Deterministic** (``run_tool_call_accuracy_check`` in
  ``aixeval.judge.runner``): no LLM. Compares the actual tool
  calls (extracted from the raw dump) against a declared, structured
  ``gold_tool_calls`` list: precision/recall F1 over ``{tool, args}``
  pairs, the same multiset-overlap idiom ``aixeval.scorecard.f1``
  already uses for text tokens. Reproducible: the same input always
  gives the same score.
- **Judge-based** (``run_tool_call_accuracy_judge``): an LLM reads the
  trajectory, a one-sentence declared ``gold_tool_plan``, and the
  output, and rates correctness. Reuses the same trajectory extraction
  ``trajectory_optimality`` already built. Cheaper to author gold data
  for (one sentence, not a structured list), at the cost of the same
  LLM-judge non-determinism already documented elsewhere in this
  toolkit.

Both write the same field, ``tool_call_accuracy_judge``, on ``[0, 1]``.
"""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "tool_call_accuracy"
# (trajectory, gold_tool_plan, output) -> correctness on [0, 1].
# This module does not call a model or compare tool calls itself. It
# grades a stored or injected verdict.
ToolCallAccuracyJudge = Callable[[str, str, str], float]


def score_tool_call_accuracy(
    record: dict[str, Any],
    judge: ToolCallAccuracyJudge | None = None,
) -> dict[str, Any]:
    """Score tool-call accuracy from a verdict on ``[0, 1]``.

    Prefers ``tool_call_accuracy_judge`` already on the record --
    written by either ``run_tool_call_accuracy_check`` (deterministic)
    or ``run_tool_call_accuracy_judge`` (LLM-based). If that field is
    missing and ``judge`` is passed, calls
    ``judge(trajectory, gold_tool_plan, output)``. Missing or invalid
    verdicts are N/A, never a silent 0.0.
    """
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("cancelled")

    if "tool_call_accuracy_judge" in record and record["tool_call_accuracy_judge"] is not None:
        return _from_verdict(record["tool_call_accuracy_judge"])

    trajectory, trajectory_ok = _optional_text(record, "trajectory")
    gold_plan, gold_plan_ok = _optional_text(record, "gold_tool_plan")
    output, output_ok = _optional_text(record, "output")
    if judge is None or not trajectory_ok or not gold_plan_ok or not output_ok or trajectory is None:
        return _na("missing_evidence")
    if gold_plan is None:
        return _na("no_gold")
    if output is None:
        return _na("missing_output")
    return _from_verdict(judge(trajectory, gold_plan, output))


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


def print_tool_call_accuracy_examples() -> None:
    """Print how a stored verdict becomes a raw score and grade."""
    print("tool_call_accuracy asks: were the right tools called with the right args?")
    print("  tool_call_accuracy_judge = a verdict on [0, 1], from either mechanism above.")
    print("  This function only grades that verdict. It does not compare tool calls itself.")
    print("  A scored row also gets a 0-4 grade from the declared higher-is-better table.")
    print("  No verdict, or a bad number, is na. No grade on N/A.")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("Verdict says fully correct tool use.", {"tool_call_accuracy_judge": 1.0}),
        ("Verdict says a required tool was skipped.", {"tool_call_accuracy_judge": 0.5}),
        (
            "No verdict on the record.",
            {"output": "68F", "trajectory": "1. get_weather -- Paris", "gold_tool_plan": "call get_weather"},
        ),
        (
            "Case was cancelled (for example an audit stop).",
            {"tool_call_accuracy_judge": 1.0, "cancelled": True},
        ),
    )
    for story, record in examples:
        result = score_tool_call_accuracy(record)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_tool_call_accuracy_examples()
