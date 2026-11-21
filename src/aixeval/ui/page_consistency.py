"""AIXeval "Consistency" page: rerun several demo scenarios, band the run-to-run spread.

Picks up where the Evaluate page's single-run flow stops: this one
needs *many* records (several problems, each judged more than once),
so it has its own page rather than a mode switch on Evaluate.
``aixeval.cross_run_consistency`` does the actual aggregation; this
page is UI glue only, same split as every other page in this app.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from aixeval.bundle import build_run_bundle
from aixeval.cross_run_consistency import ProblemRuns, score_cross_run_consistency
from aixeval.judge.live import LIVE_JUDGE_METRICS, extract_live_grounding_evidence, run_live_judges
from aixeval.scorecard import build_profile, score_metrics
from aixeval.scorecard.profile import label_for_band
from aixeval.ui.components import build_selected_judge, selected_judge_label
from aixeval.ui.scenarios import discover_scenarios, ingest_scenario, read_trace_payload


MIN_SCENARIOS = 2
MIN_RERUNS = 2
DEFAULT_RERUNS = 3
MAX_RERUNS = 10


def _eligible_metric_ids() -> tuple[str, ...]:
    """Live-judge metrics this page offers: any of them can show run-to-run spread."""
    return tuple(sorted(LIVE_JUDGE_METRICS))


def _run_consistency_check(scenario_ids: list[str], metric_id: str, reruns: int) -> None:
    """Rerun each scenario's judge ``reruns`` times, then aggregate the spread."""
    state = st.session_state
    scenarios, _problems = discover_scenarios(state["scenarios_dir"])
    by_id = {s.id: s for s in scenarios}
    client = build_selected_judge()

    problem_runs: list[ProblemRuns] = []
    run_detail: dict[str, Any] = {}
    skipped: list[str] = []

    for scenario_id in scenario_ids:
        scenario = by_id[scenario_id]
        result = ingest_scenario(scenario)[0]
        bundles = []
        raw_scores = []
        for _ in range(reruns):
            updated, _rationale, errors = run_live_judges(
                result.record,
                trajectory=result.trajectory,
                image_path=str(scenario.image_path) if scenario.image_path else None,
                selected_ids={metric_id},
                client=client,
                evidence=extract_live_grounding_evidence(read_trace_payload(scenario), result.spans),
            )
            if metric_id in errors:
                raw_scores.append(f"error: {errors[metric_id]}")
                continue
            scored = score_metrics(updated)
            by_metric = {row["metric_id"]: row for row in scored}
            row = by_metric.get(metric_id, {})
            if row.get("status") != "scored":
                raw_scores.append(f"n/a: {row.get('na_reason', 'not scored')}")
                continue
            raw_scores.append(row["raw_score"])
            profile = build_profile(scored)
            bundles.append(build_run_bundle(result, scored, profile, judge=f"{type(client).__name__} {client.model}"))
        run_detail[scenario_id] = raw_scores
        if len(bundles) >= MIN_RERUNS:
            problem_runs.append(ProblemRuns(scenario_id, bundles))
        else:
            skipped.append(scenario_id)

    state["consistency_run_detail"] = run_detail
    if not problem_runs:
        state["consistency_result"] = None
        state["consistency_error"] = (
            "No scenario got enough usable runs to score. "
            f"{metric_id} may not apply to the scenarios you picked -- see the run detail below."
        )
        return

    state["consistency_error"] = None
    if skipped:
        state["consistency_error"] = f"Skipped (too few usable runs): {', '.join(skipped)}"
    state["consistency_result"] = score_cross_run_consistency(problem_runs, metric_id=metric_id)


def render_consistency_page() -> None:
    """Pick scenarios + a judge metric, rerun each live, and band the spread."""
    st.title("Cross-run consistency")
    st.caption(
        "This toolkit scores one record at a time; this page is the one place it looks across many. "
        "Pick several demo scenarios and a judge-backed metric, rerun each scenario live more than once, "
        "and see one variance band for the whole set -- not per record."
    )

    scenarios, load_problems = discover_scenarios(st.session_state["scenarios_dir"])
    if load_problems:
        st.error(f"Could not load scenarios: {'; '.join(load_problems)}")
        return
    with_trajectory = [s for s in scenarios if s.trace_path.suffix != ".eval"]
    if len(with_trajectory) < MIN_SCENARIOS:
        st.info("Need at least two demo scenarios to run a consistency check.")
        return

    labels = {s.id: f"{s.domain} / {s.title}" for s in with_trajectory}
    selected_ids = st.multiselect(
        "Scenarios (at least 2)",
        options=list(labels),
        format_func=lambda sid: labels[sid],
        key="consistency_scenario_ids",
    )

    cols = st.columns([2, 1])
    metric_id = cols[0].selectbox(
        "Judge-backed metric to test for consistency",
        options=_eligible_metric_ids(),
        key="consistency_metric_id",
        help="Any live-judge metric: a judge's verdict can vary call to call even on an unchanged input.",
    )
    reruns = cols[1].number_input(
        "Reruns per scenario",
        min_value=MIN_RERUNS,
        max_value=MAX_RERUNS,
        value=DEFAULT_RERUNS,
        key="consistency_reruns",
    )

    st.caption(
        f"Will make {len(selected_ids) * reruns if selected_ids else 0} live judge call(s) on "
        f"{selected_judge_label()}. Switch the judge in the sidebar."
    )
    if st.button(
        "Run consistency check",
        type="primary",
        disabled=len(selected_ids) < MIN_SCENARIOS,
        icon=":material/repeat:",
    ):
        with st.spinner(f"Rerunning {len(selected_ids)} scenario(s) x {reruns} on {selected_judge_label()}..."):
            try:
                _run_consistency_check(list(selected_ids), metric_id, int(reruns))
            except Exception as exc:  # noqa: BLE001 (surfaced to the user, never crashes the app)
                st.session_state["consistency_result"] = None
                st.session_state["consistency_error"] = str(exc)

    if len(selected_ids) < MIN_SCENARIOS:
        st.caption("Pick at least two scenarios to enable this button.")

    if st.session_state["consistency_error"]:
        st.warning(st.session_state["consistency_error"])

    result = st.session_state["consistency_result"]
    if result is not None:
        _render_result(result, labels)


def _render_result(result: dict[str, Any], labels: dict[str, str]) -> None:
    """Show one scored (or honestly na) cross_run_consistency result."""
    st.divider()
    if result["status"] != "scored":
        st.error(f"Could not score: {result.get('na_reason', 'unknown')}")
        return

    grade = result["grade"]
    cols = st.columns(4)
    cols[0].metric("Grade", f"{grade}/4 ({label_for_band(grade)})")
    cols[1].metric("Mean variance", f"{result['mean_variance']:.4f}")
    cols[2].metric("Problems scored", result["problems_scored"])
    cols[3].metric("Unstable share", f"{result['unstable_share']:.0%}")

    if result["bootstrap_ci"] is not None:
        low, high = result["bootstrap_ci"]
        st.caption(f"Bootstrap 95% CI over problems: ({low:.4f}, {high:.4f})")

    if result["unstable_problems"]:
        names = ", ".join(labels.get(p["problem_id"], p["problem_id"]) for p in result["unstable_problems"])
        st.warning(f"Not fully stable: {names}")
    else:
        st.success("Every scored problem's reruns landed on the exact same raw score.")

    st.subheader("Per-problem detail")
    st.dataframe(
        [
            {
                "scenario": labels.get(p["problem_id"], p["problem_id"]),
                "variance": p["variance"],
                "runs": p["runs"],
                "unstable": p["unstable"],
            }
            for p in result["per_problem"]
        ],
        hide_index=True,
        width="stretch",
    )

    if result["excluded_problems"]:
        excluded_text = ", ".join(
            f"{labels.get(e['problem_id'], e['problem_id'])} ({e['runs_available']} usable)"
            for e in result["excluded_problems"]
        )
        st.caption(f"Excluded (too few usable runs): {excluded_text}")

    run_detail = st.session_state["consistency_run_detail"]
    if run_detail:
        with st.expander("Raw per-run scores"):
            for scenario_id, scores in run_detail.items():
                st.write(f"**{labels.get(scenario_id, scenario_id)}**: {scores}")
