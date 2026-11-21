"""Cross-run consistency: one variance band across a whole test set of problems.

This toolkit scores one record at a time (``score_metrics(record) ->
...``) and has no concept anywhere else of a "test set" -- many
problems, each run more than once. ``consistency`` and
``test_retest_stability`` both grade variance within a single record
(a two-run spread, or variance across reruns of one unchanged input).
Neither aggregates across many problems. This module is the first
piece of that missing layer: given many problems, each with several
independently-produced run bundles of the same task, it computes one
variance figure and one band for the whole set -- what a harness
actually needs to report, rather than one number per problem.

Deliberately NOT a ``POSTHOC_METRICS`` entry: it doesn't fit the
``score_x(record) -> result`` shape every scorer in ``scorecard/``
uses, the same reason ``reband``/``replay``/``compare`` live at this
top level instead of under ``scorecard/``. A caller already has the
bundles (e.g. from running ``aixeval score`` on the same task several
times); this module does not run anything itself.

Grading reuses the exact ``C(0.10)`` budget and banding
``test_retest_stability`` already uses for a single record's rerun
variance, applied here to the mean of many records' variances instead
-- the same construct, one level up.

"Unstable" is a declared policy, not pinned anywhere upstream: a
problem counts as unstable if its runs did not all land on the exact
same raw score (``variance > 0``), the simplest reading of "gave
different results across runs." Flagged here, not hidden, same
discipline every other undocumented threshold in this toolkit gets.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Any

from aixeval.scorecard.profile import grade_for_raw_score


VARIANCE_BUDGET = 0.10
MIN_RUNS_PER_PROBLEM = 2
UNSTABLE_VARIANCE_EPSILON = 0.0
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_CI = 0.95


@dataclass
class ProblemRuns:
    """One problem's repeated-run bundles: the unit this module scores.

    ``bundles`` must be at least two independently-produced run
    bundles (``aixeval.bundle.build_run_bundle`` / ``read_run_bundle``)
    of the *same* task -- e.g. ``aixeval score`` run several times on
    one problem, with live judges, so non-determinism actually has a
    chance to show up. Fewer than two usable bundles excludes the
    problem from the aggregate; it is not silently dropped.
    """

    problem_id: str
    bundles: list[dict[str, Any]]


def score_cross_run_consistency(
    problems: list[ProblemRuns],
    *,
    metric_id: str,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int | None = None,
) -> dict[str, Any]:
    """Score one metric's run-to-run variance across a whole set of problems.

    For each problem, reads ``metric_id``'s ``raw_score`` from every
    bundle where that metric actually scored (N/A rows are skipped,
    same as a missing run). A problem with fewer than
    ``MIN_RUNS_PER_PROBLEM`` usable scores is excluded from the
    aggregate, listed in ``excluded_problems`` with why, never
    silently folded in as if it were stable.

    The aggregate raw score is ``clamp(mean(per-problem variance) /
    0.10, 0, 1)``, graded lower-is-better against the shared cutoff
    table -- the same ``C(0.10)`` construct ``test_retest_stability``
    uses for one record, averaged here across many.

    Parameters
    ----------
    problems
        One ``ProblemRuns`` per problem in the set.
    metric_id
        Which metric's ``raw_score`` to read from each bundle's
        ``scored_results``. Any metric id works; this module does not
        care which one, same as ``compare_bundles`` not caring which
        dimension it is diffing.
    bootstrap_samples
        How many resamples the percentile bootstrap draws. Only used
        when at least 2 problems scored; a smaller set still returns
        a (wider, noisier) interval rather than refusing to compute
        one.
    seed
        Seeds the bootstrap's random resampling for a reproducible
        interval across repeated calls on the same input. ``None``
        (the default) draws a fresh interval each call.

    Returns
    -------
    dict
        ``status`` (``"scored"`` or ``"na"``). When scored:
        ``raw_score`` (the clamped ratio used for grading),
        ``mean_variance`` (the unclamped mean, for transparency),
        ``grade``, ``problems_scored``, ``unstable_share`` (fraction
        of scored problems with ``variance > 0``), ``unstable_problems``
        (those problems' own id/variance/runs, the "only need to check
        those" list), ``per_problem`` (every scored problem's own
        id/variance/runs), ``bootstrap_ci`` (``(low, high)`` over the
        per-problem variances, or ``None`` if fewer than 2 problems
        scored), and ``excluded_problems`` (problem id, reason, and
        how many usable runs were actually found).
        When no problem has enough usable runs: ``na_reason`` plus the
        same ``excluded_problems`` list, so the caller can see why.
    """
    scored_problems: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    for problem in problems:
        scores = _usable_scores(problem, metric_id)
        if len(scores) < MIN_RUNS_PER_PROBLEM:
            excluded.append(
                {
                    "problem_id": problem.problem_id,
                    "reason": "missing_evidence",
                    "runs_available": len(scores),
                }
            )
            continue
        variance = statistics.pvariance(scores)
        scored_problems.append(
            {
                "problem_id": problem.problem_id,
                "variance": variance,
                "runs": len(scores),
                "unstable": variance > UNSTABLE_VARIANCE_EPSILON,
            }
        )

    if not scored_problems:
        return {
            "status": "na",
            "na_reason": "missing_evidence",
            "excluded_problems": excluded,
        }

    variances = [problem["variance"] for problem in scored_problems]
    mean_variance = statistics.fmean(variances)
    raw_score = min(1.0, mean_variance / VARIANCE_BUDGET)
    grade = grade_for_raw_score(raw_score, higher_is_better=False)

    unstable_problems = [problem for problem in scored_problems if problem["unstable"]]
    unstable_share = len(unstable_problems) / len(scored_problems)
    bootstrap_ci = _bootstrap_ci(variances, samples=bootstrap_samples, seed=seed)

    return {
        "status": "scored",
        "raw_score": raw_score,
        "mean_variance": mean_variance,
        "grade": grade,
        "problems_scored": len(scored_problems),
        "unstable_share": unstable_share,
        "unstable_problems": unstable_problems,
        "per_problem": scored_problems,
        "bootstrap_ci": bootstrap_ci,
        "excluded_problems": excluded,
    }


def _usable_scores(problem: ProblemRuns, metric_id: str) -> list[float]:
    """Return every bundle's scored raw_score for metric_id. Skips N/A/missing rows."""
    scores: list[float] = []
    for bundle in problem.bundles:
        rows = bundle.get("scored_results", [])
        row = next((r for r in rows if r.get("metric_id") == metric_id), None)
        if row is None or row.get("status") != "scored":
            continue
        raw = row.get("raw_score")
        if isinstance(raw, int | float) and not isinstance(raw, bool):
            scores.append(float(raw))
    return scores


def _bootstrap_ci(
    variances: list[float], *, samples: int, seed: int | None, confidence: float = DEFAULT_CI
) -> tuple[float, float] | None:
    """Percentile bootstrap CI for the mean of ``variances``, resampling by problem.

    ``None`` with fewer than 2 problems: a single point has no spread
    to resample, so an interval would be theatre, not a real estimate.
    """
    min_problems_for_ci = 2
    if len(variances) < min_problems_for_ci:
        return None
    rng = random.Random(seed)
    n = len(variances)
    means = [statistics.fmean(rng.choices(variances, k=n)) for _ in range(samples)]
    means.sort()
    tail = (1.0 - confidence) / 2.0
    low_index = max(0, int(tail * samples))
    high_index = min(samples - 1, int((1.0 - tail) * samples) - 1)
    return means[low_index], means[high_index]


def print_cross_run_consistency_example() -> None:
    """Print a small worked set: mostly stable problems, a couple that aren't."""

    def _bundle(raw_score: float) -> dict[str, Any]:
        return {"scored_results": [{"metric_id": "task_success", "status": "scored", "raw_score": raw_score}]}

    problems = [
        ProblemRuns("p1", [_bundle(1.0), _bundle(1.0), _bundle(1.0)]),
        ProblemRuns("p2", [_bundle(1.0), _bundle(1.0), _bundle(1.0)]),
        ProblemRuns("p3", [_bundle(1.0), _bundle(0.0), _bundle(1.0)]),
        ProblemRuns("p4", [_bundle(1.0)]),  # excluded: only 1 usable run
    ]
    result = score_cross_run_consistency(problems, metric_id="task_success", seed=0)
    print("4 problems declared; p4 has only one usable run and is excluded.")
    print(f"  problems_scored:  {result['problems_scored']}")
    print(f"  mean_variance:    {result['mean_variance']:.4f}")
    print(f"  raw_score/grade:  {result['raw_score']:.4f} / {result['grade']}")
    print(f"  unstable_share:   {result['unstable_share']:.2%}")
    print(f"  unstable_problems:{[p['problem_id'] for p in result['unstable_problems']]}")
    print(f"  bootstrap_ci:     {result['bootstrap_ci']}")
    print(f"  excluded_problems:{result['excluded_problems']}")


if __name__ == "__main__":
    print_cross_run_consistency_example()
