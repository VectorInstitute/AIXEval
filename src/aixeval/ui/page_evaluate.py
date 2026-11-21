"""AIXeval "Evaluate" page: load a scenario or upload a dump, pick metrics, score."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import streamlit as st

from aixeval.bundle import build_run_bundle
from aixeval.ingest.pipeline import IngestResult, ingest_any, ingest_inspect_eval
from aixeval.ingest.trace import collect_trace_input_texts
from aixeval.judge.config import DEFAULT_PROVIDER, available_providers, default_model_for, provider_from_env
from aixeval.judge.live import (
    LIVE_JUDGE_METRICS,
    extract_live_grounding_evidence,
    live_judge_is_possible,
    run_live_judges,
)
from aixeval.judge.metric_suggester import suggest_relevant_metrics
from aixeval.judge.runner import run_protected_attribute_bias_detector
from aixeval.reband import reband_profile
from aixeval.scorecard.definitions import definition_for_any_metric
from aixeval.scorecard.hard_stops import explain_hard_stops
from aixeval.scorecard.metrics import ALL_METRIC_IDS, METRIC_DIMENSION, OPT_IN_METRICS, POSTHOC_METRICS
from aixeval.scorecard.profile import (
    BAND_LABELS,
    DIMENSIONS,
    HARD_STOP_RULES,
    HIGHER_IS_BETTER_CUTOFFS,
    build_profile,
    label_for_band,
)
from aixeval.ui.components import (
    build_selected_judge,
    render_scorecard,
    render_trace,
    render_workflow_glossary,
    selected_judge_label,
)
from aixeval.ui.logic import (
    UI_DISABLED_OPT_IN,
    analysis_matches,
    derive_run_facts,
    describe_judges,
    explain_row,
    merge_bias_signal_into_declared,
    replays_judge_verdicts,
    run_deterministic_tool_call_check,
    run_hard_stop_analysis,
    score_all,
    score_selected,
)
from aixeval.ui.scenarios import (
    Scenario,
    condition_definition,
    default_scenarios_dir,
    discover_scenarios,
    ingest_scenario,
    read_trace_payload,
)


# Widget keys whose values must survive a trip to another page: Streamlit
# drops a widget's state on any run where that widget isn't rendered.
PERSISTED_WIDGET_KEYS: tuple[str, ...] = (
    "declared_text",
    "overlay_text",
    "description",
    "audit_contamination",
    "bias_signal_text",
    "replay_mode",
    "scenarios_dir",
    "judge_provider",
    "judge_model",
    *(f"chk_{metric_id}" for metric_id in ALL_METRIC_IDS),
)

_STATE_DEFAULTS: dict[str, Any] = {
    "ingest_result": None,
    "ingest_results": [],
    "ingest_payload": None,
    "ingest_nonce": 0,
    "source_label": None,
    "source_payload": None,
    "source_eval_bytes": None,
    "active_scenario": None,
    "preset_metrics": None,
    "replay_record": None,
    "replay_rationale": {},
    "replay_analysis": None,
    "image_path": None,
    "image_upload_id": None,
    "image_is_temp": False,
    "suggestions": None,
    "suggest_error": None,
    "bias_signal_candidate": None,
    "bias_signal_scanned": False,
    "bias_signal_error": None,
    "bias_signal_applied": False,
    "bias_signal_text": "",
    "load_error": None,
    "run_results": None,
    "run_record": None,
    "run_rationale": {},
    "run_errors": {},
    "run_profile": None,
    "run_explanation": None,
    "run_analysis": None,
    "run_analysis_note": None,
    "run_judge": None,
    "judge_check": None,
    "session_runs": {},
    "_checkbox_synced_for": None,
    "declared_text": "",
    "overlay_text": "",
    "description": "",
    "audit_contamination": False,
    "replay_mode": True,
    "consistency_result": None,
    "consistency_run_detail": None,
    "consistency_error": None,
}


def init_state() -> None:
    """Seed session state once; safe to call from every page."""
    for key, value in _STATE_DEFAULTS.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("scenarios_dir", str(default_scenarios_dir()))
    if "judge_provider" not in st.session_state:
        provider = provider_from_env()
        if provider not in available_providers():
            provider = DEFAULT_PROVIDER
        st.session_state["judge_provider"] = provider
        st.session_state["judge_model"] = default_model_for(provider) or ""
    for key in PERSISTED_WIDGET_KEYS:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def _clear_checkbox_state() -> None:
    for metric_id in ALL_METRIC_IDS:
        st.session_state.pop(f"chk_{metric_id}", None)


def _pretty_json(value: dict[str, Any] | None) -> str:
    return json.dumps(value, indent=2) if value else ""


def _parse_json_box(key: str, label: str) -> dict[str, Any] | None:
    text = st.session_state.get(key, "")
    if not text.strip():
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f"{label} is not valid JSON ({exc})"
        raise ValueError(msg) from exc
    if not isinstance(data, dict):
        msg = f"{label} must be a JSON object"
        raise ValueError(msg)
    return data


def _drop_temp_image() -> None:
    path = st.session_state.get("image_path")
    if st.session_state.get("image_is_temp") and path:
        Path(path).unlink(missing_ok=True)
    st.session_state["image_path"] = None
    st.session_state["image_is_temp"] = False
    st.session_state["image_upload_id"] = None


def _sync_uploaded_image(uploaded: Any) -> None:
    """Track the image uploader on every run, not only when Ingest is clicked."""
    if uploaded is None:
        if st.session_state["image_is_temp"]:
            _drop_temp_image()
        return
    upload_id = f"{uploaded.name}:{uploaded.size}"
    if st.session_state["image_upload_id"] == upload_id:
        return
    _drop_temp_image()
    suffix = Path(uploaded.name).suffix or ".jpg"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="aixeval-image-")
    with os.fdopen(fd, "wb") as handle:
        handle.write(uploaded.getvalue())
    st.session_state["image_path"] = path
    st.session_state["image_is_temp"] = True
    st.session_state["image_upload_id"] = upload_id


def _set_ingested(
    results: list[IngestResult],
    *,
    payload: dict[str, Any] | None,
    label: str,
    preset_metrics: tuple[str, ...] | None = None,
    replay_record: dict[str, Any] | None = None,
    replay_rationale: dict[str, str] | None = None,
    replay_analysis: dict[str, Any] | None = None,
    scenario_id: str | None = None,
) -> None:
    state = st.session_state
    state["ingest_results"] = results
    state["ingest_result"] = results[0]
    state["ingest_payload"] = payload
    state["source_label"] = label
    state["active_scenario"] = scenario_id
    state["preset_metrics"] = preset_metrics
    state["replay_record"] = replay_record if len(results) == 1 else None
    state["replay_rationale"] = replay_rationale or {}
    state["replay_analysis"] = replay_analysis if len(results) == 1 else None
    state["ingest_nonce"] += 1
    state.pop("ingest_epoch_index", None)
    state.pop("trace_step", None)
    state["suggestions"] = None
    state["suggest_error"] = None
    state["bias_signal_candidate"] = None
    state["bias_signal_scanned"] = False
    state["bias_signal_error"] = None
    state["bias_signal_applied"] = False
    state["bias_signal_text"] = ""
    state["run_results"] = None
    state["load_error"] = None
    _clear_checkbox_state()


def _load_scenario(scenario: Scenario) -> None:
    """Button callback: fill every input box from the scenario, then ingest it."""
    state = st.session_state
    try:
        results = ingest_scenario(scenario)
        payload = read_trace_payload(scenario)
    except (ValueError, OSError) as exc:
        state["load_error"] = f"Could not load scenario {scenario.id}: {exc}"
        return
    state["declared_text"] = _pretty_json(scenario.declared_fields)
    state["overlay_text"] = _pretty_json(scenario.overlay)
    state["description"] = scenario.description
    state["audit_contamination"] = False
    state["source_payload"] = payload
    state["source_eval_bytes"] = scenario.trace_path.read_bytes() if payload is None else None
    _drop_temp_image()
    state["image_path"] = str(scenario.image_path) if scenario.image_path else None
    _set_ingested(
        results,
        payload=payload,
        label=scenario.title,
        preset_metrics=scenario.metric_ids,
        replay_record=scenario.saved_record,
        replay_rationale=scenario.saved_rationale,
        replay_analysis=scenario.saved_analysis,
        scenario_id=scenario.id,
    )
    state["replay_mode"] = scenario.has_saved_run


def _ingest_from_inputs(*, payload: dict[str, Any] | None, eval_bytes: bytes | None) -> list[IngestResult]:
    declared = _parse_json_box("declared_text", "Declared fields")
    if eval_bytes is not None:
        return ingest_inspect_eval(eval_bytes, declared_fields=declared)
    if payload is None:
        msg = "nothing to ingest"
        raise ValueError(msg)
    overlay = _parse_json_box("overlay_text", "Multi-agent overlay")
    return [
        ingest_any(
            payload,
            declared_fields=declared,
            overlay=overlay,
            audit_contamination=bool(st.session_state["audit_contamination"]),
        )
    ]


def _ingest_upload() -> None:
    """Button callback: ingest the uploaded file with whatever is in the input boxes."""
    state = st.session_state
    uploaded = state.get("upload_file")
    if uploaded is None:
        state["load_error"] = "Upload a trace or dump first."
        return
    try:
        if uploaded.name.endswith(".eval"):
            eval_bytes: bytes | None = uploaded.getvalue()
            payload = None
        else:
            eval_bytes = None
            payload = json.loads(uploaded.getvalue())
            if not isinstance(payload, dict):
                msg = "the uploaded JSON must be an object"
                raise ValueError(msg)
        results = _ingest_from_inputs(payload=payload, eval_bytes=eval_bytes)
    except (json.JSONDecodeError, ValueError) as exc:
        state["load_error"] = f"Could not ingest: {exc}"
        return
    state["source_payload"] = payload
    state["source_eval_bytes"] = eval_bytes
    _set_ingested(results, payload=payload, label=f"Upload: {uploaded.name}")


def _reingest_current() -> None:
    """Button callback: re-ingest the loaded run with the edited input boxes."""
    state = st.session_state
    try:
        results = _ingest_from_inputs(payload=state["source_payload"], eval_bytes=state["source_eval_bytes"])
    except (json.JSONDecodeError, ValueError) as exc:
        state["load_error"] = f"Could not re-ingest: {exc}"
        return
    label = state["source_label"] or "Run"
    if not label.endswith("(edited)"):
        label = f"{label} (edited)"
    _set_ingested(results, payload=state["source_payload"], label=label, preset_metrics=state["preset_metrics"])


def _apply_bias_signal() -> None:
    """Button callback: fold the (edited) candidate signal into Declared fields.

    Does not re-ingest by itself: same two-step pattern as editing
    Declared fields by hand, so the existing "Re-ingest current run"
    button stays the one place a record actually gets rebuilt.
    """
    state = st.session_state
    text = state.get("bias_signal_text", "").strip()
    if not text:
        return
    try:
        declared = _parse_json_box("declared_text", "Declared fields")
    except ValueError as exc:
        state["load_error"] = str(exc)
        return
    state["declared_text"] = _pretty_json(merge_bias_signal_into_declared(declared, text))
    state["bias_signal_applied"] = True


GALLERY_KEY = "scenario_gallery"
_GALLERY_CSS = (
    "<style>"
    f".st-key-{GALLERY_KEY} [data-testid='stExpander'] summary p "
    # 1.25rem / 600 is Streamlit's h5; Montserrat is
    # headingFont in .streamlit/config.toml.
    "{font-size: 1.25rem; font-weight: 600; font-family: Montserrat, sans-serif;}"
    "</style>"
)


def _render_scenario_gallery() -> None:
    with st.popover("Scenarios folder", icon=":material/folder_open:"):
        st.text_input(
            "Folder to scan",
            key="scenarios_dir",
            help="Every sub-folder holding a trace.json (or trace.eval) becomes a scenario card. Optional files "
            "next to it: declared.json, overlay.json, scenario.json, saved_run.json, image.png.",
        )
    scenarios, problems = discover_scenarios(st.session_state["scenarios_dir"])
    for problem in problems:
        st.warning(problem)
    if not scenarios:
        st.info("No scenarios found in this folder.")
        return

    with st.expander("What the workflow labels mean", icon=":material/help:"):
        render_workflow_glossary()
    domains: dict[str, list[Scenario]] = {}
    for scenario in scenarios:
        domains.setdefault(scenario.domain, []).append(scenario)
    # Static CSS only (no user content): keep the domain toggles at the
    # size of the ``#####`` headings they replaced.
    st.markdown(_GALLERY_CSS, unsafe_allow_html=True)
    active_id = st.session_state["active_scenario"]
    with st.container(key=GALLERY_KEY):
        for domain, members in domains.items():
            has_active = any(scenario.id == active_id for scenario in members)
            with st.expander(f"{domain.capitalize()} ({len(members)})", expanded=has_active):
                _render_domain_cards(members, active_id)


def _render_domain_cards(members: list[Scenario], active_id: str | None) -> None:
    cols = st.columns(3)
    for i, scenario in enumerate(members):
        with cols[i % 3].container(border=True):
            badge = f":violet-badge[{scenario.condition}] " if scenario.condition else ""
            active = active_id == scenario.id
            st.markdown(
                f"{badge}**{scenario.title}**" + ("  :green-badge[loaded]" if active else ""),
                help=condition_definition(scenario.condition),
            )
            if scenario.story:
                st.caption(scenario.story)
            st.button(
                "Load" if not active else "Reload",
                key=f"load_{scenario.id}",
                on_click=_load_scenario,
                args=(scenario,),
                type="primary" if not active else "secondary",
                icon=":material/play_arrow:",
            )


def _render_upload_tab() -> None:
    st.file_uploader(
        "Trace or dump: Langfuse export, OTel OTLP, Patronus/TRAIL, tau-bench, a simple record (.json), "
        "or an Inspect AI eval log (.eval)",
        type=["json", "eval"],
        key="upload_file",
    )
    image_file = st.file_uploader(
        "Reference image (optional, for cross_modal_hallucination)", type=["jpg", "jpeg", "png"], key="image_file"
    )
    _sync_uploaded_image(image_file)
    st.caption("Fill in the run inputs below if needed, then click **Ingest uploaded file**.")


def _render_bias_signal_detector() -> None:
    """Scan button + editable suggestion, same shape as the metric suggester.

    Advisory only: a scan never writes a record field itself. Finding a
    candidate just pre-fills an editable text box; applying it only
    edits the Declared fields box above, same as typing into it by
    hand -- "Re-ingest current run" (already below) is still the one
    place a record actually gets rebuilt.
    """
    result = st.session_state["ingest_result"]
    has_trace = result is not None and result.spans is not None
    if st.button(
        "Scan for a bias-adjacent signal",
        disabled=not has_trace,
        icon=":material/search:",
        help=(
            "Any ingested trace (single- or multi-agent). Asks a judge model to find a "
            "protected-attribute-adjacent detail in this trace's own input text (an injected "
            "note, a loaded aside) that an agent should not act on. Uses the judge picked in "
            f"the sidebar ({selected_judge_label()})."
        ),
    ):
        try:
            client = build_selected_judge()
            input_texts = collect_trace_input_texts(result.spans) if result and result.spans else ()
            with st.spinner(f"Asking {selected_judge_label()}..."):
                signal, attachment = run_protected_attribute_bias_detector(input_texts, client)
        except Exception as exc:  # noqa: BLE001 (surfaced to the user, never crashes the app)
            st.session_state["bias_signal_candidate"] = None
            st.session_state["bias_signal_scanned"] = True
            st.session_state["bias_signal_error"] = str(exc)
        else:
            st.session_state["bias_signal_scanned"] = True
            st.session_state["bias_signal_applied"] = False
            if attachment["status"] != "scored":
                st.session_state["bias_signal_candidate"] = None
                st.session_state["bias_signal_error"] = attachment.get("reason", "detector call failed")
            else:
                st.session_state["bias_signal_candidate"] = signal
                st.session_state["bias_signal_error"] = None
                st.session_state["bias_signal_text"] = signal or ""
    if not has_trace:
        st.caption("Load a trace to enable this (not a simple JSON dump).")

    if st.session_state["bias_signal_error"]:
        st.error(f"Could not scan for a bias signal: {st.session_state['bias_signal_error']}")
    elif st.session_state["bias_signal_scanned"] and st.session_state["bias_signal_candidate"] is None:
        st.caption("Scanned: no protected-attribute-adjacent signal found.")
    elif st.session_state["bias_signal_candidate"] is not None:
        st.text_area(
            "Candidate signal (edit before applying if needed)",
            key="bias_signal_text",
            height=80,
        )
        st.button(
            "Mark as a signal to resist",
            on_click=_apply_bias_signal,
            disabled=not st.session_state.get("bias_signal_text", "").strip(),
            icon=":material/check:",
            help="Adds should_resist_bias and protected_attribute_signal to Declared fields above.",
        )
        if st.session_state["bias_signal_applied"]:
            st.success("Added to Declared fields above. Click 'Re-ingest current run' below to apply it.")


def _render_inputs() -> None:
    loaded = st.session_state["ingest_result"] is not None
    with st.expander("Run inputs: declared fields, overlay, description", expanded=not loaded):
        st.text_area(
            "Declared fields (JSON, optional)",
            key="declared_text",
            height=160,
            placeholder=(
                '{"expected": "...", "should_refuse": false, "high_impact_in_scope": true, "confirmed": true, '
                '"gold_tool_calls": [{"tool": "cancel_order", "args": {"order_id": "#100"}}]}'
            ),
            help=(
                "Gold answers, safety flags, budgets: anything no raw trace carries. Always wins over anything "
                "auto-derived. 'gold_tool_calls' (a list of {tool, args}) enables a no-LLM tool_call_accuracy check "
                "for Langfuse and tau-bench uploads."
            ),
        )
        st.text_area(
            "Multi-agent overlay (JSON, optional; Langfuse/OTel/Patronus trace dumps only)",
            key="overlay_text",
            height=160,
            placeholder=(
                '{"gold_roles": [{"subtask": "find seats", "agent": "search_agent"}], '
                '"faults": [{"span_id": "f1", "contained": true}], "run_stalled": false, '
                '"mas_unsafe_steps": 0, "mas_total_steps": 6}'
            ),
            help=(
                "Gold roles, fault-containment verdicts, stall flag, MAS unsafe-step counts for the four "
                "multi-agent metrics. Does nothing for a simple record or an Inspect AI .eval upload."
            ),
        )
        st.checkbox(
            "Audit: cancel if the gold answer overlaps the trace's own input text",
            key="audit_contamination",
            help=(
                "Trace uploads only (single- or multi-agent). A windowed-overlap hit against the declared "
                "'expected' field sets contaminated=True and cancels the record. Skipped if you declare "
                "'contaminated' yourself."
            ),
        )
        _render_bias_signal_detector()
        st.text_input(
            "What is this run? (used only by the metric suggester)",
            key="description",
            placeholder="A single customer-support agent answered one text question. No images, no sub-agents.",
        )
        cols = st.columns(2)
        cols[0].button(
            "Ingest uploaded file",
            type="primary",
            on_click=_ingest_upload,
            disabled=st.session_state.get("upload_file") is None,
        )
        cols[1].button(
            "Re-ingest current run with these inputs",
            on_click=_reingest_current,
            disabled=st.session_state["source_payload"] is None and st.session_state["source_eval_bytes"] is None,
            help="Edits drop the scenario's saved run: judges run live on the edited record.",
        )


def _render_source_section() -> None:
    st.header("1. Choose a run")
    tab_scenarios, tab_upload = st.tabs([":material/auto_awesome: Demo scenarios", ":material/upload: Upload your own"])
    with tab_scenarios:
        _render_scenario_gallery()
    with tab_upload:
        _render_upload_tab()
    _render_inputs()
    if st.session_state["load_error"]:
        st.error(st.session_state["load_error"])

    results = st.session_state["ingest_results"]
    if len(results) > 1:
        index = st.selectbox(
            f"Epoch ({len(results)} in this file; each is a separate run of the same sample)",
            options=list(range(len(results))),
            format_func=lambda i: f"Epoch {i + 1}",
            key="ingest_epoch_index",
        )
        selected = results[index]
        if st.session_state["ingest_result"] is not selected:
            st.session_state["ingest_result"] = selected
            st.session_state["suggestions"] = None
            st.session_state["suggest_error"] = None
            st.session_state["run_results"] = None
            st.session_state["ingest_nonce"] += 1

    result = st.session_state["ingest_result"]
    if result is None:
        return
    st.success(f"Loaded: **{st.session_state['source_label']}**")
    scenario_story = _active_scenario_story()
    if scenario_story:
        st.info(scenario_story, icon=":material/lightbulb:")
    cols = st.columns(4)
    cols[0].metric("Source kind", result.source_kind)
    cols[1].metric("Spans", len(result.spans) if result.spans is not None else "n/a")
    cols[2].metric("Fields on record", len(result.record))
    cols[3].metric("Image", "yes" if st.session_state["image_path"] else "no")
    if result.contamination_evidence is not None:
        st.warning(f"Contamination audit flagged a leak and cancelled this record: {result.contamination_evidence}")
    elif st.session_state.get("audit_contamination"):
        if result.spans is None:
            st.caption("Contamination audit skipped: not a trace.")
        else:
            st.caption("Contamination audit: no leak found.")


def _active_scenario_story() -> str:
    scenario_id = st.session_state["active_scenario"]
    if scenario_id is None:
        return ""
    scenarios, _problems = discover_scenarios(st.session_state["scenarios_dir"])
    for scenario in scenarios:
        if scenario.id == scenario_id:
            return scenario.story
    return ""


def _render_suggester_section() -> None:
    result = st.session_state["ingest_result"]
    if result is None:
        return
    st.header("2. Suggest relevant metrics (optional)")
    st.caption(
        "Asks a judge model which metrics are *meaningful* for this run, separate from whether the data to "
        "score them is on hand yet. Scenarios already pre-select their metrics; this is for new runs. "
        f"Uses the judge picked in the sidebar ({selected_judge_label()})."
    )
    description = st.session_state.get("description", "")
    if st.button("Suggest relevant metrics", disabled=not description.strip(), icon=":material/psychology:"):
        try:
            with st.spinner(f"Asking {selected_judge_label()}..."):
                client = build_selected_judge()
                facts = derive_run_facts(result, image_path=st.session_state.get("image_path"))
                st.session_state["suggestions"] = suggest_relevant_metrics(description, client, **facts)
            st.session_state["suggest_error"] = None
        except Exception as exc:  # noqa: BLE001 (surfaced to the user, never crashes the app)
            st.session_state["suggestions"] = None
            st.session_state["suggest_error"] = str(exc)
    if not description.strip():
        st.caption('Type a description in "What is this run?" above to enable this button.')
    if st.session_state["suggest_error"]:
        st.error(f"Could not get suggestions: {st.session_state['suggest_error']}")
    if st.session_state["suggestions"]:
        names = ", ".join(sorted(st.session_state["suggestions"]))
        st.success(f"{len(st.session_state['suggestions'])} metric(s) flagged relevant: {names}. Pre-checked below.")


def _metric_is_possible(metric_id: str, scorable: dict[str, dict[str, Any]], result: IngestResult) -> bool:
    """Report whether this metric would actually score if "Run evaluation" ran now."""
    if metric_id in UI_DISABLED_OPT_IN:
        return False
    row = scorable.get(metric_id)
    if row is not None and row["status"] == "scored":
        return True
    if metric_id not in LIVE_JUDGE_METRICS:
        return False
    return live_judge_is_possible(
        metric_id,
        result.record,
        trajectory=result.trajectory,
        image_path=st.session_state["image_path"],
        evidence=extract_live_grounding_evidence(st.session_state.get("ingest_payload"), result.spans),
    )


def _select_all_possible(scorable: dict[str, dict[str, Any]], result: IngestResult) -> None:
    for metric_id in ALL_METRIC_IDS:
        st.session_state[f"chk_{metric_id}"] = _metric_is_possible(metric_id, scorable, result)


def _clear_selection() -> None:
    for metric_id in ALL_METRIC_IDS:
        st.session_state[f"chk_{metric_id}"] = False


def _default_checked(metric_id: str, *, scores_now: bool) -> bool:
    if metric_id in UI_DISABLED_OPT_IN:
        return False
    suggestions = st.session_state["suggestions"]
    if suggestions:
        return metric_id in suggestions
    preset = st.session_state["preset_metrics"]
    if preset is not None:
        return metric_id in preset
    return scores_now


def _render_metric_picker() -> set[str]:
    result: IngestResult | None = st.session_state["ingest_result"]
    if result is None:
        return set()
    st.header("3. Pick metrics")
    scorable = {row["metric_id"]: row for row in score_all(result.record)}
    suggestions = st.session_state["suggestions"] or {}
    preset = set(st.session_state["preset_metrics"] or ())

    fingerprint = (st.session_state["ingest_nonce"], tuple(sorted(suggestions)) if suggestions else None)
    if st.session_state["_checkbox_synced_for"] != fingerprint:
        _clear_checkbox_state()
        st.session_state["_checkbox_synced_for"] = fingerprint

    cols = st.columns([1, 1, 4])
    cols[0].button(
        "Select all possible",
        on_click=_select_all_possible,
        args=(scorable, result),
        help=(
            "Check every metric that would actually score right now: declared-field metrics that already score, "
            "plus live-judge metrics whose preconditions are met. Replaces the current selection."
        ),
    )
    cols[1].button("Clear", on_click=_clear_selection)

    groups: list[tuple[str, tuple[str, ...]]] = [
        (dim, tuple(m for m in POSTHOC_METRICS if METRIC_DIMENSION[m] == dim)) for dim in DIMENSIONS
    ]
    groups.append(("opt-in", OPT_IN_METRICS))

    selected: set[str] = set()
    for group_name, metric_ids in groups:
        # Seed each checkbox through session state only (never also via
        # value=), so "Select all possible"/presets can set it without
        # Streamlit's default-value-vs-session-state conflict.
        for metric_id in metric_ids:
            row = scorable.get(metric_id)
            st.session_state.setdefault(
                f"chk_{metric_id}",
                _default_checked(metric_id, scores_now=row is not None and row["status"] == "scored"),
            )
        checked_here = sum(1 for m in metric_ids if st.session_state[f"chk_{m}"])
        with st.expander(f"{group_name} ({checked_here}/{len(metric_ids)} selected)", expanded=False):
            for metric_id in metric_ids:
                row = scorable.get(metric_id)
                scores_now = row is not None and row["status"] == "scored"
                not_runnable = metric_id in UI_DISABLED_OPT_IN
                checked = st.checkbox(metric_id, key=f"chk_{metric_id}", disabled=not_runnable)
                caption = definition_for_any_metric(metric_id)
                if not_runnable:
                    caption += " (not available here: heavy local model/corpus download; use the CLI/notebook path)"
                elif scores_now:
                    caption += " Scores now."
                elif metric_id in LIVE_JUDGE_METRICS:
                    caption += " Needs a judge verdict (live call, or a scenario's saved one)."
                elif row is not None:
                    caption += f" N/A if run as-is: {row['na_reason']}."
                if metric_id in preset:
                    caption += " | Scenario preset."
                if metric_id in suggestions:
                    caption += f" | LLM: {suggestions[metric_id]}"
                st.caption(caption)
                if checked:
                    selected.add(metric_id)
    st.caption(f"{len(selected)} metric(s) selected.")
    return selected


def _render_reband_section(results: list[dict[str, Any]]) -> None:
    st.caption(
        "Change the grade cutoffs or hard-stop thresholds and see this run's card again. The raw scores are "
        "never recomputed and no model is called."
    )
    cut_cols = st.columns(4)
    cutoffs = tuple(
        col.slider(f"Grade {i + 1} at raw ≥", 0.0, 1.0, HIGHER_IS_BETTER_CUTOFFS[i], 0.05)
        for i, col in enumerate(cut_cols)
    )
    hs_cols = st.columns(3)
    contamination_cutoff = hs_cols[0].slider(
        "Contamination void at raw ≥", 0.0, 1.0, HARD_STOP_RULES["contamination_void"]["cutoff"], 0.05
    )
    unsafe_cutoff = hs_cols[1].slider(
        "Unsafe-action cap at raw ≥", 0.0, 1.0, HARD_STOP_RULES["unsafe_action_cap"]["cutoff"], 0.05
    )
    confirm_cutoff = hs_cols[2].slider(
        "Confirmation-gating cap at raw <", 0.0, 1.0, HARD_STOP_RULES["confirmation_gating_cap"]["cutoff"], 0.05
    )
    custom_rules = {
        "contamination_void": {**HARD_STOP_RULES["contamination_void"], "cutoff": contamination_cutoff},
        "unsafe_action_cap": {**HARD_STOP_RULES["unsafe_action_cap"], "cutoff": unsafe_cutoff},
        "confirmation_gating_cap": {**HARD_STOP_RULES["confirmation_gating_cap"], "cutoff": confirm_cutoff},
    }
    rebanded = reband_profile(results, cutoffs=cutoffs, hard_stop_rules=custom_rules)  # type: ignore[arg-type]
    render_scorecard(rebanded)


def _run_evaluation(result: IngestResult, selected_ids: set[str], *, replay: bool) -> None:
    state = st.session_state
    if replay and state["replay_record"] is not None:
        record = dict(state["replay_record"])
        rationale = dict(state["replay_rationale"])
    else:
        record = dict(result.record)
        rationale = {}
    errors: dict[str, str] = {}
    live_judge: str | None = None
    needs_live = _live_calls_needed(record, selected_ids)
    if needs_live:
        label = selected_judge_label()
        try:
            with st.spinner(
                f"Running {len(needs_live)} live judge call(s) on {label}: {', '.join(sorted(needs_live))}"
            ):
                client = build_selected_judge()
                live_judge = label
                record, live_rationale, errors = run_live_judges(
                    record,
                    trajectory=result.trajectory,
                    image_path=state["image_path"],
                    selected_ids=selected_ids,
                    client=client,
                    evidence=extract_live_grounding_evidence(state["ingest_payload"], result.spans),
                )
            rationale.update(live_rationale)
        except Exception as exc:  # noqa: BLE001 (a client build failure should not crash the app)
            errors = {metric_id: str(exc) for metric_id in needs_live}
    payload = state["ingest_payload"]
    if "tool_call_accuracy" in selected_ids and payload is not None:
        record, evidence = run_deterministic_tool_call_check(payload, record)
        if evidence:
            rationale["tool_call_accuracy"] = evidence

    rows = score_selected(record, selected_ids)
    profile = build_profile(rows)
    mode = "replay" if replay and state["replay_record"] is not None else "live"
    explanation = explain_hard_stops(rows, record)
    analysis, analysis_note = _incident_analysis(result, explanation, rows, record, rationale, live=mode == "live")
    if mode == "live" and analysis is not None:
        live_judge = live_judge or selected_judge_label()
    replayed = mode == "replay" and replays_judge_verdicts(state["replay_record"], selected_ids)
    judge = describe_judges(
        live_judge=live_judge,
        replayed=replayed,
        replay_record=state["replay_record"] if replayed else None,
    )
    state["run_judge"] = judge
    state["run_results"] = rows
    state["run_record"] = record
    state["run_rationale"] = rationale
    state["run_errors"] = errors
    state["run_profile"] = profile
    state["run_explanation"] = explanation
    state["run_analysis"] = analysis
    state["run_analysis_note"] = analysis_note
    label = f"{state['source_label']} [{mode} · {live_judge}]" if live_judge else f"{state['source_label']} [{mode}]"
    state["session_runs"] = {
        **state["session_runs"],
        label: build_run_bundle(replace(result, record=record), rows, profile, judge=judge),
    }


def _incident_analysis(
    result: IngestResult,
    explanation: dict[str, Any],
    rows: list[dict[str, Any]],
    record: dict[str, Any],
    rationale: dict[str, str],
    *,
    live: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    """Reuse a matching saved analysis on replay; call the judge model on a live run."""
    if not explanation["hard_stops"]:
        return None, None
    saved = None if live else st.session_state["replay_analysis"]
    if analysis_matches(saved, explanation):
        return saved, None
    if not live:
        return None, "No saved incident analysis matches these hard stops. Turn replay off to generate one live."
    try:
        with st.spinner(f"Hard stop fired: running the incident analysis on {selected_judge_label()}"):
            client = build_selected_judge()
            analysis, error = run_hard_stop_analysis(
                explanation, rows, record=record, spans=result.spans, rationale=rationale, client=client
            )
    except Exception as exc:  # noqa: BLE001 (a client build failure should not crash the app)
        return None, f"Incident analysis unavailable: {exc}"
    return analysis, None if error is None else f"Incident analysis failed: {error}"


def _live_calls_needed(record: dict[str, Any], selected_ids: set[str]) -> set[str]:
    return {m for m in selected_ids & LIVE_JUDGE_METRICS.keys() if LIVE_JUDGE_METRICS[m].record_field not in record}


def _render_results_section(selected_ids: set[str]) -> None:
    result: IngestResult | None = st.session_state["ingest_result"]
    if result is None:
        return
    st.header("4. Run evaluation")
    has_replay = st.session_state["replay_record"] is not None
    if has_replay:
        st.toggle(
            "Replay the scenario's saved judge verdicts (no API calls, same result every time)",
            key="replay_mode",
            help="Off: every selected judge runs live against the judge model picked below.",
        )
    replay = has_replay and bool(st.session_state["replay_mode"])
    base_record = st.session_state["replay_record"] if replay else result.record
    needs_live = _live_calls_needed(base_record, selected_ids)
    if needs_live:
        st.caption(
            f"Live judge calls on {selected_judge_label()} will run for: {', '.join(sorted(needs_live))}. "
            "Switch the judge in the sidebar."
        )
    elif selected_ids:
        st.caption("No live judge calls needed: every selected verdict is already on the record.")

    if st.button("Run evaluation", type="primary", disabled=not selected_ids, icon=":material/rocket_launch:"):
        _run_evaluation(result, selected_ids, replay=replay)

    rows = st.session_state["run_results"]
    if rows is None:
        return
    run_judge = st.session_state["run_judge"]
    if run_judge:
        st.info(f"Judge: {run_judge}")
    else:
        st.caption("Judge: none (every selected metric is deterministic).")
    for metric_id, message in st.session_state["run_errors"].items():
        st.warning(f"{metric_id}: live judge call failed ({message}). Reported N/A below.")
    profile = st.session_state["run_profile"]
    rationale = st.session_state["run_rationale"]

    tab_card, tab_metrics, tab_trace, tab_reband, tab_export = st.tabs(
        ["Scorecard", "Per-metric detail", "Trace", "Reband", "Record & export"]
    )
    with tab_card:
        render_scorecard(
            profile,
            explanation=st.session_state["run_explanation"],
            analysis=st.session_state["run_analysis"],
            analysis_note=st.session_state["run_analysis_note"],
            spans=result.spans,
            trace_key="trace_step",
        )
    with tab_metrics:
        table = [
            {
                "metric": row["metric_id"],
                "dimension": METRIC_DIMENSION.get(row["metric_id"], "opt-in"),
                "grade": row["grade"] if row["status"] == "scored" else None,
                "band": label_for_band(row["grade"]) if row["status"] == "scored" else "N/A",
                "raw": row.get("raw_score"),
            }
            for row in sorted(rows, key=lambda r: (METRIC_DIMENSION.get(r["metric_id"], "~"), r["metric_id"]))
        ]
        st.dataframe(table, hide_index=True, width="stretch")
        for row in sorted(rows, key=lambda r: r["metric_id"]):
            grade = f"grade {row['grade']}/4 ({BAND_LABELS[row['grade']]})" if row["status"] == "scored" else "N/A"
            with st.expander(f"{row['metric_id']}: {grade}"):
                st.write(explain_row(row, rationale))
    with tab_trace:
        render_trace(result.spans, key="trace_step")
    with tab_reband:
        _render_reband_section(rows)
    with tab_export:
        bundle_record = st.session_state["run_record"] or result.record
        bundle = build_run_bundle(replace(result, record=bundle_record), rows, profile, judge=run_judge)
        st.download_button(
            "Download run bundle (.json)",
            data=json.dumps(bundle, indent=2),
            file_name="run_bundle.json",
            mime="application/json",
            icon=":material/download:",
            help="The sealed record (with every judge verdict), the trace, every metric's result and the profile. "
            "Reopen it on the Compare page or with the CLI, no re-run needed.",
        )
        st.json(bundle_record, expanded=False)


def render_evaluate_page() -> None:
    """Render the page: choose a run, suggest, pick, run; top to bottom."""
    st.title("Evaluate a run")
    st.caption("Posthoc scorecard for a saved agent run: no live harness, this reads a run that already happened.")
    _render_source_section()
    _render_suggester_section()
    selected_ids = _render_metric_picker()
    _render_results_section(selected_ids)
