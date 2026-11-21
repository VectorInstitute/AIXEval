"""AIXeval "Overview" page: what the toolkit measures, and how to start a demo."""

from __future__ import annotations

import streamlit as st
from streamlit.navigation.page import StreamlitPage

from aixeval.scorecard.definitions import DIMENSION_DEFINITIONS, HARD_STOP_DEFINITIONS
from aixeval.scorecard.metrics import METRIC_DIMENSION, POSTHOC_METRICS
from aixeval.scorecard.profile import BAND_LABELS, DIMENSIONS
from aixeval.ui.components import APP_NAME, APP_TAGLINE, GRADE_COLORS, LOGO_STACKED, render_workflow_glossary
from aixeval.ui.scenarios import discover_scenarios


def render_overview_page(evaluate_page: StreamlitPage) -> None:
    """Landing page: one paragraph, the eight questions, the grade scale, hard stops."""
    head, logo = st.columns([4, 1])
    with head:
        st.title(APP_NAME)
        st.markdown(f"#### {APP_TAGLINE}")
        st.markdown(
            "AIXeval reads a saved agent run (a Langfuse, OpenTelemetry, Patronus or tau-bench trace, an Inspect "
            "AI log, or a simple record), scores it against a library of metrics, and rolls the results into an "
            "**eight-question scorecard**. Judge-backed metrics call an LLM judge; everything else is "
            "deterministic. Declared **hard stops** cap the card when a safety rule is broken, however well "
            "the task itself went."
        )
    logo.image(str(LOGO_STACKED), width="stretch")

    scenarios, _problems = discover_scenarios(st.session_state["scenarios_dir"])
    cols = st.columns(4)
    cols[0].metric("Card questions", len(DIMENSIONS))
    cols[1].metric("Metrics", len(POSTHOC_METRICS))
    cols[2].metric("Demo scenarios", len(scenarios))
    cols[3].metric("Hard stops", len(HARD_STOP_DEFINITIONS))

    if st.button("Start with a demo scenario", type="primary", icon=":material/play_arrow:"):
        st.switch_page(evaluate_page)

    st.subheader("The eight questions")
    question_cols = st.columns(4)
    for i, dimension in enumerate(DIMENSIONS):
        count = sum(1 for metric_id in POSTHOC_METRICS if METRIC_DIMENSION[metric_id] == dimension)
        with question_cols[i % 4].container(border=True):
            st.markdown(f"**{dimension.capitalize()}**")
            st.caption(DIMENSION_DEFINITIONS[dimension])
            st.caption(f"{count} metric(s)")

    left, right = st.columns(2)
    with left:
        st.subheader("Grade scale")
        for grade, label in enumerate(BAND_LABELS):
            st.markdown(
                f"<span style='display:inline-block;width:0.9em;height:0.9em;border-radius:3px;"
                f"background:{GRADE_COLORS[grade]};margin-right:0.5em'></span>**{grade}** {label}",
                unsafe_allow_html=True,
            )
        st.caption(
            "Unanswered questions are omitted from the bars and listed as chips under the card. "
            "That is missing evidence, not a failing grade."
        )
    with right:
        st.subheader("Hard stops")
        for stop, definition in HARD_STOP_DEFINITIONS.items():
            st.markdown(f"**{stop}**: {definition}")

    st.subheader("Scenario workflows")
    st.caption("Each demo scenario is labelled with the agent workflow it exercises.")
    render_workflow_glossary()
