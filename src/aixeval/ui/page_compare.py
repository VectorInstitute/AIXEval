"""AIXeval "Compare" page: two runs side by side, per dimension and per metric."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from aixeval.bundle import BUNDLE_SCHEMA_VERSION, compare_bundles
from aixeval.scorecard.profile import DIMENSIONS, build_profile
from aixeval.ui.components import comparison_chart, render_scorecard
from aixeval.ui.logic import compare_rows, describe_judges, replays_judge_verdicts, score_selected
from aixeval.ui.scenarios import Scenario, discover_scenarios


def _scenario_run(scenario: Scenario) -> dict[str, Any]:
    """Score a scenario's saved run on its own preset metrics; no model call."""
    selected = set(scenario.metric_ids) if scenario.metric_ids is not None else set()
    rows = score_selected(scenario.saved_record or {}, selected)
    replayed = replays_judge_verdicts(scenario.saved_record, selected)
    judge = describe_judges(live_judge=None, replayed=replayed, replay_record=scenario.saved_record)
    return {"scored_results": rows, "profile": build_profile(rows), "judge": judge}


def _judge_caption(run: dict[str, Any]) -> str:
    if "judge" not in run:
        return "Judge: not recorded in this bundle."
    return f"Judge: {run['judge']}" if run["judge"] else "Judge: none (every scored metric is deterministic)."


def _uploaded_bundles() -> dict[str, dict[str, Any]]:
    bundles: dict[str, dict[str, Any]] = {}
    for uploaded in st.session_state.get("compare_uploads") or []:
        try:
            bundle = json.loads(uploaded.getvalue())
        except json.JSONDecodeError as exc:
            st.warning(f"{uploaded.name}: not valid JSON ({exc})")
            continue
        if not isinstance(bundle, dict) or bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION:
            st.warning(f"{uploaded.name}: not a run bundle (schema_version {BUNDLE_SCHEMA_VERSION} expected)")
            continue
        bundles[f"Bundle: {uploaded.name}"] = bundle
    return bundles


def _available_runs() -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    scenarios, _problems = discover_scenarios(st.session_state["scenarios_dir"])
    for scenario in scenarios:
        if scenario.has_saved_run:
            runs[f"Scenario: {scenario.title}"] = _scenario_run(scenario)
    for label, bundle in st.session_state["session_runs"].items():
        runs[f"This session: {label}"] = bundle
    runs.update(_uploaded_bundles())
    return runs


def _default_pair(labels: list[str]) -> tuple[int, int]:
    """Open on the strongest demo pair if present: hiring confirmed vs unconfirmed."""
    wanted = (
        "Scenario: Rejection email: waits for confirmation",
        "Scenario: Rejection email: sent without confirmation",
    )
    if all(label in labels for label in wanted):
        return labels.index(wanted[0]), labels.index(wanted[1])
    return 0, min(1, len(labels) - 1)


def render_compare_page() -> None:
    """Pick two runs and show their cards, a grouped chart, and per-metric deltas."""
    st.title("Compare two runs")
    st.caption(
        "Scenario saved runs, runs from this session (Evaluate page), or uploaded run bundles. "
        "Nothing is re-scored with a model here."
    )
    with st.expander("Upload run bundles", icon=":material/upload:"):
        st.file_uploader("Run bundle (.json)", type=["json"], accept_multiple_files=True, key="compare_uploads")

    runs = _available_runs()
    if len(runs) < 2:
        st.info("Need at least two runs to compare: load scenarios, run an evaluation, or upload bundles.")
        return
    labels = list(runs)
    default_a, default_b = _default_pair(labels)
    cols = st.columns(2)
    label_a = cols[0].selectbox("Run A", labels, index=default_a, key="compare_a")
    label_b = cols[1].selectbox("Run B", labels, index=default_b, key="compare_b")
    run_a, run_b = runs[label_a], runs[label_b]
    profile_a, profile_b = run_a["profile"], run_b["profile"]

    verdict = compare_bundles(run_a, run_b, max_grade_drift=0)
    changed = [diff["dimension"] for diff in verdict["dimension_diffs"]]
    if changed:
        st.warning(f"Grades differ on: {', '.join(changed)}.")
    else:
        st.success("Every dimension both runs scored has the same grade.")

    st.altair_chart(comparison_chart(profile_a, profile_b, "A", "B"), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("A")
        st.caption(label_a)
        st.caption(_judge_caption(run_a))
        render_scorecard(profile_a)
    with right:
        st.subheader("B")
        st.caption(label_b)
        st.caption(_judge_caption(run_b))
        render_scorecard(profile_b)
    if run_a.get("judge") and run_b.get("judge") and run_a["judge"] != run_b["judge"]:
        st.info("These runs were judged differently, so a grade gap may come from the judge, not the agent.")

    st.subheader("Per-metric grades")
    lined_up = compare_rows(run_a["scored_results"], run_b["scored_results"])
    if not lined_up:
        st.info("Neither run scored any metric.")
        return
    st.dataframe(
        [
            {
                "metric": row["metric_id"],
                "A": row["grade_a"],
                "B": row["grade_b"],
                "B - A": row["delta"],
            }
            for row in lined_up
        ],
        hide_index=True,
        width="stretch",
    )
    only_one = [d for d in DIMENSIONS if (profile_a.get(d) is None) != (profile_b.get(d) is None)]
    if only_one:
        st.caption(f"Scored in only one run (not compared): {', '.join(only_one)}.")
