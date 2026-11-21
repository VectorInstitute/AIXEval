"""AIXeval Streamlit app: overview, evaluate a run (scenario or upload), compare runs.

Run from the repo root (so ``.streamlit/config.toml``'s Vector theme is picked up):
    uv run --group ui streamlit run src/aixeval/ui/app.py

Demo scenarios are read from ``fixtures/scenarios/`` (override with
``$AIXEVAL_SCENARIOS_DIR`` or the folder picker on the Evaluate page);
see ``aixeval.ui.scenarios`` for the folder layout.

All scoring/ingest/judge logic lives in the library proper or in
``aixeval.ui.logic`` / ``aixeval.ui.scenarios`` (no Streamlit
import, unit tested). The page modules are widgets only.
"""

from __future__ import annotations

import streamlit as st

from aixeval.ui.components import APP_NAME, FAVICON, render_branding, render_judge_picker
from aixeval.ui.page_compare import render_compare_page
from aixeval.ui.page_consistency import render_consistency_page
from aixeval.ui.page_evaluate import init_state, render_evaluate_page
from aixeval.ui.page_overview import render_overview_page


def main() -> None:
    """Configure the page, seed state, and route to the selected page."""
    st.set_page_config(page_title=APP_NAME, page_icon=str(FAVICON), layout="wide")
    init_state()
    render_branding()
    render_judge_picker()

    evaluate = st.Page(render_evaluate_page, title="Evaluate", icon=":material/fact_check:", url_path="evaluate")
    compare = st.Page(render_compare_page, title="Compare", icon=":material/compare_arrows:", url_path="compare")
    consistency = st.Page(
        render_consistency_page, title="Consistency", icon=":material/repeat:", url_path="consistency"
    )
    overview = st.Page(
        lambda: render_overview_page(evaluate),
        title="Overview",
        icon=":material/home:",
        url_path="overview",
        default=True,
    )
    st.navigation([overview, evaluate, compare, consistency], position="top").run()


if __name__ == "__main__":
    main()
