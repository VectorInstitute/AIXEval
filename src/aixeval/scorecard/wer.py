"""wer raw score: word error rate clipped to [0, 1], then a lower-is-better grade."""

from __future__ import annotations

from typing import Any

import jiwer

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "wer"
# Tier 1 (library-backed): our own scorer, jiwer does the edit-distance math.
# jiwer.wer is unbounded above 1.0 (a hypothesis much longer than the
# reference can exceed it), clipped to [0, 1] before grading, same
# clipping discipline as latency_vs_budget / step_efficiency.


def score_wer(record: dict[str, Any]) -> dict[str, Any]:
    """Score word error rate of ``output`` against ``expected`` (gold transcript).

    Reads ``expected`` (reference) and ``output`` (hypothesis): the
    same fields ``task_success`` reads, not a separate reference field.
    Missing gold or output is N/A, never a silent 0.0.
    """
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("cancelled")

    expected, expected_ok = _optional_text(record, "expected")
    output, output_ok = _optional_text(record, "output")
    if not expected_ok or not output_ok:
        return _na("missing_evidence")
    if expected is None:
        return _na("no_gold")
    if output is None:
        return _na("missing_output")

    raw_score = min(1.0, max(0.0, jiwer.wer(expected, output)))
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
    return apply_grade(
        {"metric_id": METRIC_ID, "status": "scored", "raw_score": raw_score},
        higher_is_better=False,
    )


def _na(reason: str) -> dict[str, Any]:
    return {"metric_id": METRIC_ID, "status": "na", "na_reason": require_na_reason(reason)}


def print_wer_examples() -> None:
    """Print how expected/output become a WER raw score and grade."""
    print("wer asks: how many word edits separate output from expected?")
    print("  expected = gold transcript. output = what the model said.")
    print("  jiwer.wer, clipped to [0, 1] -> raw_score. Lower is better.")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("Exact match text.", {"expected": "the cat sat on the mat", "output": "the cat sat on the mat"}),
        ("One word dropped.", {"expected": "the cat sat on the mat", "output": "the cat sat on mat"}),
        ("Unrelated text (would exceed 1.0, clipped).", {"expected": "a", "output": "a completely different sentence"}),
        ("No gold answer on the record.", {"output": "yes"}),
        ("Gold exists, but the model gave no answer.", {"expected": "yes"}),
    )
    for story, record in examples:
        result = score_wer(record)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_wer_examples()
