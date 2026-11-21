"""pairwise_preference raw score: a judge verdict on output vs. a baseline.

Deliberately not named ``win_rate``: "LLM-judge win-rate / Elo" is a
cross-system aggregate (quartile rank within a consortium pool)
rolled up from many pairwise verdicts, not a per-record score. This
metric is the per-record building block one level below that: one
judged preference between ``output`` and a declared
``baseline_output``, on ``[0, 1]``. Rolling many of these into an
actual win-rate or Elo rating is future work, not attempted here.

Position-bias-corrected, per ``judge_pairwise_preference`` (Zheng
et al., NeurIPS 2023): the verdict already reflects an average over
both presentation orders, not a single, order-sensitive judgment.
"""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "pairwise_preference"
# A judge (human or model) rates output vs. baseline_output on [0, 1].
# This module does not call a model. It grades a stored or injected verdict.
PairwisePreferenceJudge = Callable[[str, str], float]


def score_pairwise_preference(
    record: dict[str, Any],
    judge: PairwisePreferenceJudge | None = None,
) -> dict[str, Any]:
    """Score a pairwise preference verdict on ``[0, 1]``.

    Prefers ``pairwise_preference_judge`` already on the record. If
    that field is missing and ``judge`` is passed, calls
    ``judge(output, baseline_output)``. The toolkit does not ship a
    model client. Missing or invalid verdicts are N/A, never a
    silent 0.0.
    """
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("cancelled")

    if "pairwise_preference_judge" in record and record["pairwise_preference_judge"] is not None:
        return _from_verdict(record["pairwise_preference_judge"])

    output, output_ok = _optional_text(record, "output")
    baseline, baseline_ok = _optional_text(record, "baseline_output")
    if judge is None or not output_ok or not baseline_ok:
        return _na("missing_evidence")
    if baseline is None:
        return _na("no_gold")
    if output is None:
        return _na("missing_output")
    return _from_verdict(judge(output, baseline))


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


def print_pairwise_preference_examples() -> None:
    """Print how a judge verdict becomes a raw score and grade."""
    print("pairwise_preference asks: is output better than a declared baseline_output?")
    print("  A judge (model or human) returns pairwise_preference_judge on [0, 1].")
    print("  This function only grades that verdict. It does not call a model.")
    print("  1.0 = output clearly better. 0.0 = baseline clearly better. 0.5 = tie.")
    print("  No verdict, or a bad number, is na. No grade on N/A.")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("Judge says output is clearly better.", {"pairwise_preference_judge": 1.0}),
        ("Judge says the baseline is clearly better.", {"pairwise_preference_judge": 0.0}),
        ("Judge says output has a slight edge (raw 0.6 -> grade 3).", {"pairwise_preference_judge": 0.6}),
        ("No baseline, no judge, and no stored verdict.", {"output": "Paris."}),
        (
            "A baseline exists, but still no judge and no stored verdict.",
            {"output": "Paris.", "baseline_output": "The capital of France is Paris."},
        ),
        (
            "Case was cancelled (for example an audit stop).",
            {"pairwise_preference_judge": 1.0, "cancelled": True},
        ),
    )
    for story, record in examples:
        result = score_pairwise_preference(record)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_pairwise_preference_examples()
