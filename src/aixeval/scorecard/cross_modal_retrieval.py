"""cross_modal_retrieval raw score: did the correct match land in the top-k.

Literature: Recall@k, standard cross-modal retrieval evaluation
(e.g. the MS-COCO retrieval split, Flickr30k): the fraction of
queries whose correct match falls within the top k retrieved
candidates.

Declared, not computed: this toolkit does not run retrieval itself
(no embedding index, no nearest-neighbor search). The harness that
performed the retrieval declares where the correct match landed
(``retrieval_rank``, 1-indexed) and the cutoff it was judged against
(``retrieval_k``); this module only checks ``rank <= k``. Same
discipline as ``cost_per_query``/``latency_vs_budget``: a declared
fact, not a live computation.
"""

from __future__ import annotations

from typing import Any

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "cross_modal_retrieval"


def score_cross_modal_retrieval(record: dict[str, Any]) -> dict[str, Any]:
    """Score whether the correct match's declared rank falls within the declared top-k.

    Reads ``retrieval_rank`` (int >= 1, 1-indexed) and ``retrieval_k``
    (int >= 1). Raw score is 1.0 if ``retrieval_rank <= retrieval_k``,
    else 0.0. Higher is better. Missing or invalid counts are N/A,
    never a silent 0.0.
    """
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("cancelled")

    rank, rank_ok = _positive_int(record, "retrieval_rank")
    k, k_ok = _positive_int(record, "retrieval_k")
    if not rank_ok or not k_ok or rank is None or k is None:
        return _na("missing_evidence")
    return _scored(1.0 if rank <= k else 0.0)


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


def print_cross_modal_retrieval_examples() -> None:
    """Print how a declared rank and k become a raw score and grade."""
    print("cross_modal_retrieval asks: did the correct match land in the top-k?")
    print("  retrieval_rank = where the correct match actually landed (1-indexed)")
    print("  retrieval_k    = the declared cutoff")
    print("  rank <= k -> raw_score 1.0 (grade 4). rank > k -> raw_score 0.0 (grade 0).")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("Correct match ranked 1st, k=5.", {"retrieval_rank": 1, "retrieval_k": 5}),
        ("Correct match ranked 5th, k=5.", {"retrieval_rank": 5, "retrieval_k": 5}),
        ("Correct match ranked 9th, k=5.", {"retrieval_rank": 9, "retrieval_k": 5}),
        ("No rank on the record.", {"retrieval_k": 5}),
        (
            "Case was cancelled (for example an audit stop).",
            {"retrieval_rank": 1, "retrieval_k": 5, "cancelled": True},
        ),
    )
    for story, record in examples:
        result = score_cross_modal_retrieval(record)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_cross_modal_retrieval_examples()
