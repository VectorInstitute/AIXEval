"""pass_hat_k_agent raw score: probability that k runs of a problem all succeed.

Literature: adapted from Chen, Tworek, Jun, et al., *Evaluating Large
Language Models Trained on Code*, 2021 (arXiv:2107.03374, verified via
search, not guessed) -- the same pass@k family ``pass_at_k_agent``
cites -- but a different question. The paper's own pass@k is the
probability that *at least one* of k sampled attempts succeeds:
``1 - C(n-c, k) / C(n, k)``. This metric asks the stricter reliability
question instead: the probability that *all* k sampled attempts
succeed: ``C(c, k) / C(n, k)``. An agent that solves a task most of
the time but not reliably scores low here even though it would score
near 1 on ordinary pass@k.

Computed, not declared, unlike ``pass_at_k_agent``: that metric reads
an already-decided ``solved_within_budget`` outcome; this one computes
the combinatorial probability itself from two declared per-problem
counts, ``pass_hat_k_attempts`` (n, independent runs of the same
problem under identical conditions) and ``pass_hat_k_successes`` (c,
how many of those runs succeeded). k defaults to 5, capped at n when
fewer runs were declared, matching the usual multi-trial setup (as in
tau-bench). This toolkit still scores one record (one problem) at a
time: averaging many records' raw scores recovers a pass^k-style rate
across a suite, the same discipline ``pass_at_k_agent``'s own
docstring already describes for its construct.
"""

from __future__ import annotations

from math import comb
from typing import Any

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "pass_hat_k_agent"
DEFAULT_K = 5


def score_pass_hat_k_agent(record: dict[str, Any]) -> dict[str, Any]:
    """Score the probability that k independent runs of this problem all succeed.

    Reads ``pass_hat_k_attempts`` (int >= 1, n) and
    ``pass_hat_k_successes`` (int >= 0, c, must not exceed n). k is
    ``min(5, n)``. Raw score is ``comb(c, k) / comb(n, k)``. Higher is
    better: 1.0 means every declared attempt succeeded (so any k of
    them all succeed); 0.0 means fewer than k attempts succeeded, so no
    k-subset of all-successes exists. Missing or invalid counts are
    N/A, never a silent 0.0.
    """
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("cancelled")

    attempts, attempts_ok = _positive_int(record, "pass_hat_k_attempts")
    successes, successes_ok = _nonneg_int(record, "pass_hat_k_successes")
    if not attempts_ok or not successes_ok or attempts is None or successes is None:
        return _na("missing_evidence")
    if successes > attempts:
        return _na("missing_evidence")

    k = min(DEFAULT_K, attempts)
    raw_score = comb(successes, k) / comb(attempts, k)
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


def print_pass_hat_k_agent_examples() -> None:
    """Print how declared per-problem (n, c) counts become a raw score and grade."""
    print("pass_hat_k_agent asks: if you ran this problem k more times, would ALL k succeed?")
    print("  raw_score = comb(c, k) / comb(n, k), k = min(5, n). Higher is better.")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("5 successes in 5 runs: every run succeeded.", {"pass_hat_k_attempts": 5, "pass_hat_k_successes": 5}),
        ("4 successes in 5 runs: not fully reliable.", {"pass_hat_k_attempts": 5, "pass_hat_k_successes": 4}),
        ("Only 3 runs declared: k caps at 3, not 5.", {"pass_hat_k_attempts": 3, "pass_hat_k_successes": 3}),
        ("No successes at all.", {"pass_hat_k_attempts": 5, "pass_hat_k_successes": 0}),
        ("No attempt count on the record.", {"pass_hat_k_successes": 3}),
        ("Successes exceed attempts: invalid declared data.", {"pass_hat_k_attempts": 3, "pass_hat_k_successes": 5}),
        (
            "Case was cancelled (for example an audit stop).",
            {"pass_hat_k_attempts": 5, "pass_hat_k_successes": 5, "cancelled": True},
        ),
    )
    for story, record in examples:
        result = score_pass_hat_k_agent(record)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_pass_hat_k_agent_examples()
