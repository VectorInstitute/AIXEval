"""vqa_accuracy raw score: VQAv2-style soft accuracy over multiple annotator answers.

Literature: Goyal et al., Making the V in VQA Matter, CVPR 2017
(VQAv2; common alternatives include GQA and MMBench). VQAv2's own
convention: an answer counts as fully
correct once at least 3 of 10 independent human annotators gave that
same answer, with partial credit below that:

    Acc = min(1, (count of annotators giving the model's answer) / 3)

This needs the *set* of annotator answers, not a single gold string
-- a single ``expected`` can't express "3 of 10 agreed," so this
reads a declared ``gold_answers: list[str]`` instead (one entry per
annotator who gave that answer; duplicates are meaningful and
expected, not deduplicated).

Declared simplification, flagged honestly: real VQAv2 scoring also
normalizes both the prediction and each gold answer (lowercasing,
stripping punctuation and articles, number-word normalization) before
comparing. This module normalizes lowercase/punctuation/whitespace
only (the same normalization ``f1``/``bleu`` already use elsewhere in
this toolkit), not VQAv2's full official normalization script, the
same declared-variant spirit as ``bleu``/``rouge`` not reproducing
their own official scripts exactly either.
"""

from __future__ import annotations

import re
from typing import Any

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "vqa_accuracy"
# VQAv2's own convention: 3 of 10 annotators agreeing counts as fully correct.
AGREEMENT_DENOMINATOR = 3
_PUNCTUATION_RE = re.compile(r"[^\w\s]")


def score_vqa_accuracy(record: dict[str, Any]) -> dict[str, Any]:
    """Score ``output`` against ``gold_answers`` using VQAv2's soft-accuracy formula.

    Reads ``gold_answers`` (one declared string per annotator who gave
    that answer) and ``output``. Raw score is ``min(1, matches / 3)``,
    where ``matches`` counts how many ``gold_answers`` entries equal
    ``output`` after normalization. Higher is better. Missing gold or
    output is N/A, never a silent 0.0.
    """
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("cancelled")

    gold_answers, gold_ok = _optional_str_list(record, "gold_answers")
    output, output_ok = _optional_text(record, "output")
    if not gold_ok or not output_ok:
        return _na("missing_evidence")
    if gold_answers is None:
        return _na("no_gold")
    if output is None:
        return _na("missing_output")

    normalized_output = _normalize(output)
    matches = sum(1 for answer in gold_answers if _normalize(answer) == normalized_output)
    raw_score = min(1.0, matches / AGREEMENT_DENOMINATOR)
    return _scored(raw_score)


def _normalize(text: str) -> str:
    return " ".join(_PUNCTUATION_RE.sub(" ", text.lower()).split())


def _optional_str_list(record: dict[str, Any], key: str) -> tuple[list[str] | None, bool]:
    """Return a non-empty string list and whether the field type is usable."""
    if key not in record or record[key] is None:
        return None, True
    value = record[key]
    if not isinstance(value, list) or not value:
        return None, False
    if not all(isinstance(item, str) and item for item in value):
        return None, False
    return value, True


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


def print_vqa_accuracy_examples() -> None:
    """Print how output/gold_answers become a VQA soft-accuracy raw score and grade."""
    print("vqa_accuracy asks: how many annotators gave the same answer as output?")
    print("  raw_score = min(1, matches / 3): VQAv2's own soft-accuracy convention.")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("All annotators agree, output matches.", {"output": "yellow", "gold_answers": ["yellow"] * 10}),
        (
            "3 of 10 annotators said this: fully correct.",
            {"output": "yellow", "gold_answers": ["yellow"] * 3 + ["gold"] * 7},
        ),
        (
            "Only 1 of 10 annotators said this: partial credit.",
            {"output": "yellow", "gold_answers": ["yellow"] + ["gold"] * 9},
        ),
        ("No annotator gave this answer.", {"output": "blue", "gold_answers": ["yellow"] * 10}),
        ("No gold_answers on the record.", {"output": "yellow"}),
        (
            "Case was cancelled (for example an audit stop).",
            {"output": "yellow", "gold_answers": ["yellow"] * 10, "cancelled": True},
        ),
    )
    for story, record in examples:
        result = score_vqa_accuracy(record)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_vqa_accuracy_examples()
