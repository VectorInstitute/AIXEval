"""Page-level AIXeval checks through Streamlit's AppTest: no browser, no judge calls.

AppTest can't drive a file uploader or switch to a function-backed page,
so these tests render each page function directly and go through the
demo-scenario buttons, which exercise the same ingest/score path an
upload would. Every run here is a replay: no provider is ever called.
"""

from pathlib import Path

import pytest


pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402


APP_PATH = Path(__file__).resolve().parents[1] / "src" / "aixeval" / "ui" / "app.py"
UNCONFIRMED = "hiring/rejection_email_unconfirmed"
CONFIRMED = "hiring/rejection_email_confirmed"


def _evaluate_and_compare() -> None:
    import streamlit as st  # noqa: PLC0415

    from aixeval.ui.components import render_judge_picker  # noqa: PLC0415
    from aixeval.ui.page_compare import render_compare_page  # noqa: PLC0415
    from aixeval.ui.page_evaluate import init_state, render_evaluate_page  # noqa: PLC0415

    init_state()
    render_judge_picker()
    if st.session_state.get("_test_page") == "compare":
        render_compare_page()
    else:
        render_evaluate_page()


def _fresh_evaluate_page() -> AppTest:
    return AppTest.from_function(_evaluate_and_compare, default_timeout=60).run()


def _load_and_run(at: AppTest, scenario_id: str) -> AppTest:
    at.button(key=f"load_{scenario_id}").click().run()
    next(b for b in at.button if b.label == "Run evaluation").click().run()
    return at


def test_the_full_app_opens_on_the_aixeval_overview() -> None:
    at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not at.exception
    assert [title.value for title in at.title] == ["AIXeval"]
    assert at.selectbox(key="judge_provider").value in {
        "anthropic",
        "gemini",
        "openai",
        "vllm",
    }


def test_switching_the_sidebar_judge_updates_the_model_to_that_provider_default() -> None:
    from aixeval.judge.config import default_model_for  # noqa: PLC0415

    at = _fresh_evaluate_page()
    at.selectbox(key="judge_provider").select("anthropic").run()
    assert not at.exception
    assert at.selectbox(key="judge_provider").value == "anthropic"
    assert at.text_input(key="judge_model").value == default_model_for("anthropic")


def _domain_toggles(at: AppTest) -> dict[str, bool]:
    """Map each domain toggle's label to whether it is open."""
    from aixeval.ui.scenarios import default_scenarios_dir, discover_scenarios  # noqa: PLC0415

    domains = {s.domain.capitalize() for s in discover_scenarios(default_scenarios_dir())[0]}
    return {e.label: e.proto.expanded for e in at.expander if e.label.split(" (")[0] in domains}


def test_each_domain_is_one_collapsed_toggle_holding_its_scenarios() -> None:
    at = _fresh_evaluate_page()
    assert not at.exception
    toggles = _domain_toggles(at)
    assert "Hiring (11)" in toggles
    assert len(toggles) >= 4
    assert not any(toggles.values())
    hiring = next(e for e in at.expander if e.label == "Hiring (11)")
    assert f"load_{UNCONFIRMED}" in {b.key for b in hiring.button}


def test_loading_a_scenario_opens_only_its_domain_toggle() -> None:
    at = _fresh_evaluate_page()
    at.button(key=f"load_{UNCONFIRMED}").click().run()
    toggles = _domain_toggles(at)
    assert [label for label, opened in toggles.items() if opened] == ["Hiring (11)"]


def test_loading_a_scenario_fills_every_input_and_preselects_its_metrics() -> None:
    at = _fresh_evaluate_page()
    at.button(key=f"load_{UNCONFIRMED}").click().run()
    assert not at.exception
    assert '"confirmed": false' in at.text_area(key="declared_text").value
    assert "hiring agent" in at.text_input(key="description").value
    checked = {c.key.removeprefix("chk_") for c in at.checkbox if c.key and c.key.startswith("chk_") and c.value}
    assert {"confirmation_gating", "unsafe_action", "task_success", "grounding"} <= checked
    assert at.toggle(key="replay_mode").value is True
    assert any("No live judge calls needed" in caption.value for caption in at.caption)


def test_replaying_the_unconfirmed_hiring_run_trips_both_hard_stops() -> None:
    at = _load_and_run(_fresh_evaluate_page(), UNCONFIRMED)
    assert not at.exception
    profile = at.session_state["run_profile"]
    assert (profile["capability"], profile["safety"], profile["oversight"]) == (0, 0, 0)
    assert set(profile["applied_hard_stops"]) == {"unsafe_action_cap", "confirmation_gating_cap"}
    assert any("Hard stop applied" in error.value for error in at.error)
    assert not at.warning


def test_a_replayed_hard_stop_shows_its_saved_incident_analysis() -> None:
    at = _load_and_run(_fresh_evaluate_page(), UNCONFIRMED)
    assert not at.exception
    analysis = at.session_state["run_analysis"]
    assert analysis is not None
    assert analysis["hard_stops"] == ["unsafe_action_cap", "confirmation_gating_cap"]
    assert at.session_state["run_explanation"]["capped_questions"]
    assert any(e.label.startswith("Incident analysis") for e in at.expander)
    first = analysis["responsible_steps"][0]
    at.button(key=f"jump_trace_step_{first}").click().run()
    assert not at.exception
    inspector = at.selectbox(key="trace_step")
    assert inspector.index is not None
    assert inspector.options[inspector.index].startswith(first)


def test_metric_checkboxes_are_never_given_a_default_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Seed checkboxes via session state only, so presets never conflict with value=.

    Streamlit logs that conflict at most once per process, so asserting on
    the log would pass vacuously; this spies on the rule check instead.
    """
    from streamlit.elements.lib import policies  # noqa: PLC0415

    defaults_seen: list[tuple[str, object]] = []
    original = policies.check_session_state_rules

    def spy(default_value: object, key: str | None, writes_allowed: bool = True) -> None:
        if key is not None and key.startswith("chk_") and default_value is not None:
            defaults_seen.append((key, default_value))
        original(default_value, key, writes_allowed)

    monkeypatch.setattr(policies, "check_session_state_rules", spy)
    at = _fresh_evaluate_page()
    at.button(key=f"load_{CONFIRMED}").click().run()
    next(b for b in at.button if b.label == "Select all possible").click().run()
    assert not at.exception
    assert any(c.value for c in at.checkbox if c.key and c.key.startswith("chk_"))
    assert defaults_seen == []


def test_compare_page_opens_on_the_hiring_pair_and_flags_the_difference() -> None:
    at = _fresh_evaluate_page()
    at.session_state["_test_page"] = "compare"
    at.run()
    assert not at.exception
    assert at.selectbox(key="compare_a").value == "Scenario: Rejection email: waits for confirmation"
    assert at.selectbox(key="compare_b").value == "Scenario: Rejection email: sent without confirmation"
    assert any("capability" in warning.value and "safety" in warning.value for warning in at.warning)


def test_a_session_run_becomes_available_on_the_compare_page() -> None:
    at = _load_and_run(_fresh_evaluate_page(), CONFIRMED)
    at.session_state["_test_page"] = "compare"
    at.run()
    assert not at.exception
    assert "This session: Rejection email: waits for confirmation [replay]" in at.selectbox(key="compare_a").options
