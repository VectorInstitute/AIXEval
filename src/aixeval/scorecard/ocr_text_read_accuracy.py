"""ocr_text_read_accuracy raw score: 1 - character error rate, higher-is-better.

Literature: Character Error Rate (CER), standard OCR/text-recognition
evaluation metric (edit distance at the character level over a gold
transcript). Common per-dataset benchmarks include TextVQA, DocVQA,
and ChartQA.

Tier 1 (library-backed), no new dependency: ``jiwer`` (already a
dependency for ``wer``) implements ``jiwer.cer`` directly: the same
edit-distance math as ``jiwer.wer``, just tokenized per character
instead of per word.

No new image I/O: the OCR step (reading text out of an image) already
happened upstream of this toolkit; ``output`` is the text that was
read, ``expected`` is the gold transcript. This module compares two
strings, the same shape as ``wer``/``task_success``, not a separate
image-aware metric.
"""

from __future__ import annotations

from typing import Any

import jiwer

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "ocr_text_read_accuracy"
# jiwer.cer is unbounded above 1.0 (a hypothesis much longer than the
# reference can exceed it), clipped before the 1 - CER flip, same
# clipping discipline as wer / latency_vs_budget / step_efficiency.


def score_ocr_text_read_accuracy(record: dict[str, Any]) -> dict[str, Any]:
    """Score ``1 - character error rate`` of ``output`` against ``expected``.

    Reads ``expected`` (gold transcript) and ``output`` (what the OCR
    step read): the same fields ``wer``/``task_success`` read, not a
    separate transcript field. Missing gold or output is N/A, never a
    silent 0.0.
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

    cer = min(1.0, max(0.0, jiwer.cer(expected, output)))
    raw_score = min(1.0, max(0.0, 1.0 - cer))
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


def print_ocr_text_read_accuracy_examples() -> None:
    """Print how expected/output become an OCR raw score and grade."""
    print("ocr_text_read_accuracy asks: how much of the in-image text was read correctly?")
    print("  expected = gold transcript. output = what the OCR step read.")
    print("  raw_score = 1 - jiwer.cer(expected, output), clipped to [0, 1]. Higher is better.")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("Exact match text.", {"expected": "INVOICE #4471", "output": "INVOICE #4471"}),
        ("One character misread.", {"expected": "INVOICE #4471", "output": "INVOICE #447I"}),
        (
            "Unrelated text (CER would exceed 1.0, clipped).",
            {"expected": "A", "output": "a completely different string"},
        ),
        ("No gold transcript on the record.", {"output": "INVOICE #4471"}),
        ("Gold exists, but OCR produced no output.", {"expected": "INVOICE #4471"}),
    )
    for story, record in examples:
        result = score_ocr_text_read_accuracy(record)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_ocr_text_read_accuracy_examples()
