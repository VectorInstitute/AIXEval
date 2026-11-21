"""pass_at_k_agent raw score: was the task solved within its declared attempt budget.

Literature: Chen, Tworek, Jun, et al., *Evaluating Large Language
Models Trained on Code*, 2021 (arXiv:2107.03374, verified via search,
not guessed), introduces pass@k and its unbiased combinatorial
estimator: given n sampled attempts with c correct among them, pass@k
= 1 - C(n-c, k) / C(n, k).

Declared simplification, not the paper's own estimator: that formula
needs n >= k independent samples per task and a corpus-level
aggregation step. This toolkit scores one record at a time (the
``score_x(record) -> ...`` shape every metric here uses), so this
module reduces the construct to a binary per-record outcome: did this
task get solved within its declared k-attempt budget. Aggregating
many records' 0/1 outcomes (e.g. via the mean) recovers a pass@k-style
rate across a suite, the same way ``refusal``/``abstention`` rates are
recovered from many per-record 0/1 verdicts elsewhere in this toolkit,
not the unbiased estimator itself. Flagged here, not hidden, same
discipline as ``modality_robustness``'s banding-formula simplification
and ``pairwise_preference``'s two-call approximation of token-
probability averaging.

Distinct from code-execution pass@k (needing a real code-execution/
test-suite environment this toolkit does not have; correctly out of
scope). ``pass_at_k_agent`` scores a
behavioral outcome some other harness already determined, not a code
test run.
"""

from __future__ import annotations

from typing import Any

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "pass_at_k_agent"


def score_pass_at_k_agent(record: dict[str, Any]) -> dict[str, Any]:
    """Score whether the task was solved within its declared attempt budget.

    Reads ``attempt_budget`` (int >= 1, the declared k) and
    ``solved_within_budget`` (bool). Raw score is 1.0 if solved, else
    0.0. Higher is better. ``attempt_budget`` is required even though
    it does not enter the arithmetic directly: without it, a bare
    ``solved_within_budget`` flag has no declared k to be read against,
    the same discipline ``interruptibility`` applies to
    ``stop_signal_sent``. Missing or invalid fields are N/A, never a
    silent 0.0.
    """
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("cancelled")

    budget, budget_ok = _positive_int(record, "attempt_budget")
    if not budget_ok or budget is None:
        return _na("missing_evidence")

    solved, solved_ok = _optional_bool(record, "solved_within_budget")
    if not solved_ok or solved is None:
        return _na("missing_evidence")
    return _scored(1.0 if solved else 0.0)


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


def _optional_bool(record: dict[str, Any], key: str) -> tuple[bool | None, bool]:
    """Return a bool and whether the field type is usable."""
    if key not in record or record[key] is None:
        return None, True
    value = record[key]
    if not isinstance(value, bool):
        return None, False
    return value, True


def _scored(raw_score: float) -> dict[str, Any]:
    return apply_grade({"metric_id": METRIC_ID, "status": "scored", "raw_score": raw_score})


def _na(reason: str) -> dict[str, Any]:
    return {"metric_id": METRIC_ID, "status": "na", "na_reason": require_na_reason(reason)}


def print_pass_at_k_agent_examples() -> None:
    """Print how a declared attempt budget and outcome become a raw score and grade."""
    print("pass_at_k_agent asks: was the task solved within its declared k-attempt budget?")
    print("  attempt_budget        = the declared k")
    print("  solved_within_budget  = did the task get solved within that budget")
    print("  Solved -> raw_score 1.0 (grade 4). Not solved -> raw_score 0.0 (grade 0).")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("Solved within a budget of 3 tries.", {"attempt_budget": 3, "solved_within_budget": True}),
        ("Exhausted a budget of 3 tries, never solved.", {"attempt_budget": 3, "solved_within_budget": False}),
        ("No attempt budget on the record.", {"solved_within_budget": True}),
        ("Budget present, but no outcome flag.", {"attempt_budget": 3}),
        (
            "Case was cancelled (for example an audit stop).",
            {"attempt_budget": 3, "solved_within_budget": True, "cancelled": True},
        ),
    )
    for story, record in examples:
        result = score_pass_at_k_agent(record)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_pass_at_k_agent_examples()
