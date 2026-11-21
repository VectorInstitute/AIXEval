"""bleu raw score: sacrebleu sentence BLEU on [0, 1], then a higher-is-better grade."""

from __future__ import annotations

from typing import Any

import sacrebleu

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "bleu"
# Tier 1 (library-backed): our own scorer, sacrebleu does the n-gram math.
# sacrebleu's .score is on [0, 100]; we divide by 100 for our [0, 1] scale.


def score_bleu(record: dict[str, Any]) -> dict[str, Any]:
    """Score n-gram precision of ``output`` against ``expected`` (reference text).

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

    result = sacrebleu.sentence_bleu(output, [expected])
    # sacrebleu can overshoot 100.0 by float rounding (e.g. 100.00000000000004
    # on some exact matches); clip before grading, same as wer's clip.
    raw_score = min(1.0, max(0.0, result.score / 100.0))
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


def print_bleu_examples() -> None:
    """Print how expected/output become a BLEU raw score and grade."""
    print("bleu asks: how much n-gram overlap does output have with expected?")
    print("  expected = reference text. output = what the model said.")
    print("  sacrebleu.sentence_bleu, score / 100 -> raw_score on [0, 1].")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("Exact match text.", {"expected": "the cat sat on the mat", "output": "the cat sat on the mat"}),
        ("Close paraphrase.", {"expected": "the cat sat on the mat", "output": "the cat is sitting on the mat"}),
        ("Unrelated text.", {"expected": "the cat sat on the mat", "output": "quarterly revenue rose 3 percent"}),
        ("No gold answer on the record.", {"output": "yes"}),
        ("Gold exists, but the model gave no answer.", {"expected": "yes"}),
    )
    for story, record in examples:
        result = score_bleu(record)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_bleu_examples()
