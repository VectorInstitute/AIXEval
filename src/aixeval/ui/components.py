"""Streamlit rendering helpers shared by the AIXeval pages: branding, card, trace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import altair as alt
import streamlit as st

from aixeval.ingest.trace import Span
from aixeval.judge.config import available_providers, build_client, default_model_for
from aixeval.judge.provider import Provider
from aixeval.scorecard.definitions import DIMENSION_DEFINITIONS, HARD_STOP_DEFINITIONS
from aixeval.scorecard.profile import BAND_LABELS, DIMENSIONS
from aixeval.ui.logic import judge_label, span_rows
from aixeval.ui.scenarios import QUALIFIER_DEFINITIONS, WORKFLOW_DEFINITIONS


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_HORIZONTAL = ASSETS_DIR / "vector-logo-dark.svg"
LOGO_STACKED = ASSETS_DIR / "vector-logo-stacked.png"
FAVICON = ASSETS_DIR / "favicon.svg"

APP_NAME = "AIXeval"
APP_TAGLINE = "Posthoc agent evaluation scorecard"

# Vector brand palette, ordered from grade 0 (Unacceptable) to 4 (Excellent).
GRADE_COLORS: tuple[str, ...] = ("#E8553A", "#FF9E00", "#48C0D9", "#7ED3B2", "#1DB47F")
NA_COLOR = "#EDEDED"
# Grade 0 would draw a zero-length bar; a short stub keeps "Unacceptable" visible.
_ZERO_GRADE_STUB = 0.12
RUN_COLORS: tuple[str, str] = ("#EB088A", "#313CFF")

_KIND_MARKERS = {"agent": "●", "llm": "◆", "tool": "■"}


def render_branding() -> None:
    """Corner logo plus a sidebar footer; call once per run, before navigation."""
    st.logo(str(LOGO_HORIZONTAL), size="large", icon_image=str(FAVICON))
    st.sidebar.caption("Vector Institute · AI Engineering")


def selected_judge_label() -> str:
    """Return this session's live judge, e.g. ``vllm · gemma-3-12b-it``."""
    provider = str(st.session_state["judge_provider"])
    model = str(st.session_state.get("judge_model") or "").strip() or default_model_for(provider) or "(no model set)"
    return judge_label(provider, model)


def build_selected_judge() -> Provider:
    """Build a client for the sidebar's current provider + model."""
    return build_client(st.session_state["judge_provider"], st.session_state["judge_model"])


def _on_judge_provider_change() -> None:
    st.session_state["judge_model"] = default_model_for(st.session_state["judge_provider"]) or ""
    st.session_state["judge_check"] = None


def render_judge_picker() -> None:
    """Sidebar: show and switch the live judge. ``.env`` only supplies the defaults."""
    with st.sidebar:
        st.subheader("Judge")
        st.caption("Used for live calls. Replay still uses a scenario's saved verdicts.")
        st.selectbox(
            "Provider",
            available_providers(),
            key="judge_provider",
            on_change=_on_judge_provider_change,
            help="Starts from JUDGE_PROVIDER in .env. A change here lasts for this browser session only.",
        )
        st.text_input(
            "Model",
            key="judge_model",
            placeholder="the model id your server is serving",
            help="Starts from that provider's JUDGE_MODEL_* in .env, else its built-in default. "
            "vllm has no default: name the model your server was started with.",
        )
        label = selected_judge_label()
        try:
            client = build_selected_judge()
        except Exception as exc:  # noqa: BLE001 (a missing key or model is shown, never crashes the app)
            st.warning(f"{label} can't be used: {exc}")
            return
        if st.button(
            "Test connection",
            icon=":material/network_check:",
            help="Sends one tiny prompt. Uses 2048 tokens so a reasoning model "
            "Some reasoning models can finish thinking and still emit a word.",
        ):
            try:
                with st.spinner(f"Calling {label}..."):
                    reply = client.complete("Reply with the single word: ok", max_tokens=2048)
                st.session_state["judge_check"] = (label, True, reply.strip()[:80])
            except Exception as exc:  # noqa: BLE001 (any provider failure is shown to the user)
                st.session_state["judge_check"] = (label, False, f"{type(exc).__name__}: {exc}")
        check = st.session_state.get("judge_check")
        if check is not None and check[0] == label:
            _, ok, detail = check
            if ok:
                st.success(f"{label} answered: {detail!r}")
            else:
                st.error(f"{label} failed: {detail}")


def render_workflow_glossary() -> None:
    """List each scenario workflow type and qualifier with its one-sentence meaning."""
    for name, definition in WORKFLOW_DEFINITIONS.items():
        st.markdown(f":violet-badge[{name}] {definition}")
    st.caption("Qualifiers after a comma:")
    for name, definition in QUALIFIER_DEFINITIONS.items():
        st.markdown(f"**{name}**: {definition}")


def _profile_chart_rows(profile: dict[str, Any], run: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for dimension in DIMENSIONS:
        grade: int | None = profile.get(dimension)
        row: dict[str, Any] = {"dimension": dimension, "question": DIMENSION_DEFINITIONS[dimension], "run": run or ""}
        if grade is None:
            row.update(label=f"{dimension}  N/A", grade=None, bar=4, color=NA_COLOR, band="Not scored", text="")
        else:
            row.update(
                label=f"{dimension}  {grade}/4",
                grade=grade,
                bar=max(grade, _ZERO_GRADE_STUB),
                color=GRADE_COLORS[grade],
                band=BAND_LABELS[grade],
                text=f"{grade} {BAND_LABELS[grade]}",
            )
        rows.append(row)
    return rows


def scorecard_chart(profile: dict[str, Any]) -> alt.Chart | None:
    """Draw one bar per scored dimension.

    Unscored questions are omitted here and listed as chips under the card.
    """
    rows = [row for row in _profile_chart_rows(profile) if row["grade"] is not None]
    if not rows:
        return None
    base = alt.Chart(alt.InlineData(values=rows)).encode(
        y=alt.Y(
            "label:N", sort=[row["label"] for row in rows], title=None, axis=alt.Axis(labelFontSize=13, labelLimit=220)
        ),
        x=alt.X(
            "bar:Q",
            scale=alt.Scale(domain=[0, 4.6]),
            axis=alt.Axis(values=[0, 1, 2, 3, 4], title="Grade (0 Unacceptable – 4 Excellent)"),
        ),
        tooltip=[
            alt.Tooltip("dimension:N", title="Dimension"),
            alt.Tooltip("question:N", title="Question"),
            alt.Tooltip("band:N", title="Band"),
        ],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=22).encode(color=alt.Color("color:N", scale=None))
    labels = base.mark_text(align="left", dx=6, fontSize=12, color="#222222").encode(text="text:N")
    # Step height keeps one band per scored row; a fixed pixel height collapses
    # when Streamlit stretches a short scored-only card.
    return (bars + labels).properties(height=alt.Step(38))


def comparison_chart(profile_a: dict[str, Any], profile_b: dict[str, Any], name_a: str, name_b: str) -> alt.Chart:
    """Draw grouped bars per dimension for two runs; an N/A dimension has no bar."""
    rows = [
        row
        for profile, name in ((profile_a, name_a), (profile_b, name_b))
        for row in _profile_chart_rows(profile, run=name)
        if profile.get(row["dimension"]) is not None
    ]
    base = alt.Chart(alt.InlineData(values=rows)).encode(
        y=alt.Y("dimension:N", sort=list(DIMENSIONS), title=None, axis=alt.Axis(labelFontSize=13, labelLimit=220)),
        yOffset=alt.YOffset("run:N", sort=[name_a, name_b]),
        x=alt.X("bar:Q", scale=alt.Scale(domain=[0, 4.6]), axis=alt.Axis(values=[0, 1, 2, 3, 4], title="Grade")),
        tooltip=[
            alt.Tooltip("run:N", title="Run"),
            alt.Tooltip("dimension:N", title="Dimension"),
            alt.Tooltip("grade:Q", title="Grade"),
            alt.Tooltip("band:N", title="Band"),
        ],
    )
    bars = base.mark_bar(cornerRadiusEnd=4).encode(
        color=alt.Color(
            "run:N",
            scale=alt.Scale(domain=[name_a, name_b], range=list(RUN_COLORS)),
            legend=alt.Legend(orient="top", title=None),
        )
    )
    labels = base.mark_text(align="left", dx=6, fontSize=12, color="#222222").encode(text="text:N")
    return (bars + labels).properties(height=56 * len(DIMENSIONS))


def render_scorecard(
    profile: dict[str, Any],
    *,
    explanation: dict[str, Any] | None = None,
    analysis: dict[str, Any] | None = None,
    analysis_note: str | None = None,
    spans: list[Span] | None = None,
    trace_key: str | None = None,
) -> None:
    """Hard-stop banner (if any), scored bars only, chips for unanswered questions.

    With ``explanation`` (from ``explain_hard_stops``) the banner also says
    what made each stop fire and what it capped; ``analysis`` adds the
    model's incident write-up, and ``analysis_note`` says why there is none.
    ``spans`` plus ``trace_key`` let a cited step jump the trace inspector.
    Unscored dimensions are listed as chips under the chart, not grey N/A bars.
    """
    hard_stops = list(profile.get("applied_hard_stops") or ())
    if hard_stops:
        reasons = "\n".join(f"- **{stop}**: {HARD_STOP_DEFINITIONS.get(stop, '')}" for stop in hard_stops)
        st.error(f"**Hard stop applied.** The card is capped regardless of other scores.\n\n{reasons}")
        if explanation and explanation["hard_stops"]:
            _render_hard_stop_facts(explanation)
            _render_incident_analysis(analysis, analysis_note, spans, trace_key)
    chart = scorecard_chart(profile)
    if chart is not None:
        st.altair_chart(chart, width="stretch")
    else:
        st.info("No dimensions scored on this run.")
    unscored = [dimension for dimension in DIMENSIONS if profile.get(dimension) is None]
    if unscored:
        chips = " ".join(f":gray-badge[{dimension} · not asked]" for dimension in unscored)
        st.markdown(chips)
    filled = sum(1 for dimension in DIMENSIONS if profile.get(dimension) is not None)
    if unscored:
        st.caption(
            f"Showing {filled} scored question{'s' if filled != 1 else ''}. "
            "The chips are outside this run's evidence, not errors or failing grades."
        )
    else:
        st.caption(f"All {filled} dimensions scored.")


def _render_hard_stop_facts(explanation: dict[str, Any]) -> None:
    st.markdown("**Why the card was capped**")
    for stop in explanation["hard_stops"]:
        raw = stop["raw_score"]
        evidence = ", ".join(f"`{field} = {json.dumps(value)}`" for field, value in stop["evidence_fields"].items())
        st.markdown(
            f"- `{stop['hard_stop']}` fired: {stop['rule']} (raw score {raw if raw is not None else 'n/a'}), "
            f"so {stop['effect']}." + (f" Driven by {evidence}." if evidence else "")
        )
    capped = explanation["capped_questions"]
    if capped:
        st.dataframe(
            [
                {
                    "question": question,
                    "without the hard stop": change["before"],
                    "on the card": "N/A" if change["after"] is None else change["after"],
                }
                for question, change in capped.items()
            ],
            hide_index=True,
            width="stretch",
        )


def _render_incident_analysis(
    analysis: dict[str, Any] | None,
    note: str | None,
    spans: list[Span] | None,
    trace_key: str | None,
) -> None:
    with st.expander("Incident analysis (model explanation; it does not change any grade)", expanded=True):
        if analysis is None:
            st.info(note or "No incident analysis for this run.")
            return
        st.markdown(f"**Root cause.** {analysis['root_cause']}")
        steps = analysis["responsible_steps"]
        if steps:
            st.markdown("**Responsible steps.** " + ", ".join(f"`{step}`" for step in steps))
            _render_step_jumps(steps, spans, trace_key)
        if analysis["contributing_factors"]:
            st.markdown("**Contributing factors.**\n" + "\n".join(f"- {f}" for f in analysis["contributing_factors"]))
        st.markdown(f"**What a passing run would do.** {analysis['what_would_pass']}")
        st.caption(
            f"Confidence {analysis['confidence']:.2f} · {analysis.get('model', 'unknown model')} · "
            f"{analysis.get('prompt_version', '')}"
        )


def _render_step_jumps(steps: list[str], spans: list[Span] | None, trace_key: str | None) -> None:
    if not spans or trace_key is None:
        return
    index_by_id = {row["id"]: index for index, row in enumerate(span_rows(spans))}
    targets = [step for step in steps if step in index_by_id]
    if not targets:
        return
    cols = st.columns(len(targets))
    for col, step in zip(cols, targets, strict=True):
        col.button(
            f"Inspect {step}",
            key=f"jump_{trace_key}_{step}",
            on_click=_jump_to_step,
            args=(trace_key, index_by_id[step]),
            help="Selects this step in the Trace tab's inspector.",
        )


def _jump_to_step(trace_key: str, index: int) -> None:
    st.session_state[trace_key] = index
    st.toast("Step selected: open the Trace tab to see it.")


def _step_name(row: dict[str, Any]) -> str:
    """Name a step: the agent for an agent span, else the tool or model name."""
    if row["kind"] == "agent" and row["agent"]:
        return str(row["agent"])
    return str(row["name"] or row["agent"] or row["id"])


def render_trace(spans: list[Span] | None, *, key: str) -> None:
    """Span tree as a table, an optional timeline, and an input/output inspector."""
    if not spans:
        st.info("This run has no trace: it was a simple record, so there are no steps to show.")
        return
    rows = span_rows(spans)
    agents = sorted({row["agent"] for row in rows if row["agent"]})
    tools = sorted({row["name"] for row in rows if row["kind"] == "tool" and row["name"]})
    cols = st.columns(3)
    cols[0].metric("Spans", len(rows))
    cols[1].metric("Agents", len(agents))
    cols[2].metric("Distinct tools", len(tools))

    table = [
        {
            "step": f"{'    ' * row['depth']}{_KIND_MARKERS.get(row['kind'], '·')} {_step_name(row)}",
            "kind": row["kind"],
            "agent": row["agent"] or "",
            "status": row["status"] or "",
            "seconds": row["duration_s"],
        }
        for row in rows
    ]
    st.dataframe(table, hide_index=True, width="stretch")

    timed = [row for row in rows if row["start"] and row["end"]]
    if len(timed) >= 2:
        step_order = [f"{i + 1:02d} {_step_name(row)}" for i, row in enumerate(timed)]
        timeline = [
            {"step": step, "kind": row["kind"], "start": row["start"], "end": row["end"]}
            for step, row in zip(step_order, timed, strict=True)
        ]
        chart = (
            alt.Chart(alt.InlineData(values=timeline))
            .mark_bar(cornerRadius=3)
            .encode(
                y=alt.Y("step:N", sort=step_order, title=None),
                x=alt.X("start:T", title="Time"),
                x2="end:T",
                color=alt.Color(
                    "kind:N",
                    scale=alt.Scale(domain=["agent", "llm", "tool"], range=["#8A25C9", "#313CFF", "#EB088A"]),
                    legend=alt.Legend(orient="top", title=None),
                ),
                tooltip=["step:N", "kind:N", "start:T", "end:T"],
            )
            .properties(height=max(120, 24 * len(timeline)))
        )
        with st.expander("Timeline", expanded=False):
            st.altair_chart(chart, width="stretch")

    labels = [f"{row['id']} · {row['kind']} · {_step_name(row)}" for row in rows]
    choice = st.selectbox("Inspect a step", options=list(range(len(rows))), format_func=lambda i: labels[i], key=key)
    picked = rows[choice]
    left, right = st.columns(2)
    left.markdown("**Input**")
    left.text(picked["input"] or "(empty)")
    right.markdown("**Output**")
    right.text(picked["output"] or "(empty)")
