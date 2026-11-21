"""f1 raw score: token-overlap precision/recall F1 on [0, 1].

Literature: Rajpurkar et al., SQuAD: 100,000+ Questions for Machine
Comprehension of Text, EMNLP 2016: token-F1 is the standard QA-span
evaluation convention introduced there.

Tier 1 (algorithmic/reference), but no new dependency: token overlap
is simple enough to implement directly: lowercase, strip
punctuation, split on whitespace, multiset (Counter) intersection. A
declared normalization choice, not a reproduction of SQuAD's exact
official script (which also strips articles), same spirit as
ROUGE-L being one declared variant among several.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "f1"
_PUNCTUATION_RE = re.compile(r"[^\w\s]")


def score_f1(record: dict[str, Any]) -> dict[str, Any]:
    """Score token-overlap F1 of ``output`` against ``expected``.

    Reads ``expected`` (gold) and ``output`` (hypothesis): the same
    fields ``task_success`` reads, not separate gold/predicted spans.
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

    raw_score = _token_f1(output, expected)
    return _scored(raw_score)


def _token_f1(hypothesis: str, reference: str) -> float:
    hyp_tokens = _tokenize(hypothesis)
    ref_tokens = _tokenize(reference)
    if not hyp_tokens and not ref_tokens:
        return 1.0
    if not hyp_tokens or not ref_tokens:
        return 0.0
    common = Counter(hyp_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(hyp_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _tokenize(text: str) -> list[str]:
    return _PUNCTUATION_RE.sub(" ", text.lower()).split()


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


def print_f1_examples() -> None:
    """Print how expected/output become an F1 raw score and grade."""
    print("f1 asks: how much token overlap (precision + recall) does output have with expected?")
    print("  expected = gold text. output = what the model said.")
    print("  Lowercase, strip punctuation, split on whitespace, then multiset overlap.")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("Exact match text.", {"expected": "the cat sat on the mat", "output": "the cat sat on the mat"}),
        ("Partial overlap.", {"expected": "the cat sat on the mat", "output": "a cat sat on a rug"}),
        ("Unrelated text.", {"expected": "the cat sat on the mat", "output": "quarterly revenue rose 3 percent"}),
        ("No gold answer on the record.", {"output": "yes"}),
        ("Gold exists, but the model gave no answer.", {"expected": "yes"}),
    )
    for story, record in examples:
        result = score_f1(record)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_f1_examples()
