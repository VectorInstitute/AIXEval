"""ui.scenarios: folder discovery, sidecar loading, and the shipped demo scenarios."""

import json
from pathlib import Path
from typing import Any

import pytest

from aixeval.scorecard.hard_stops import explain_hard_stops
from aixeval.scorecard.metrics import score_metrics
from aixeval.ui.logic import analysis_matches, score_selected
from aixeval.ui.scenarios import (
    QUALIFIER_DEFINITIONS,
    SCENARIOS_DIR_ENV,
    WORKFLOW_DEFINITIONS,
    condition_definition,
    default_scenarios_dir,
    discover_scenarios,
    ingest_scenario,
    load_scenario,
    read_trace_payload,
)


SHIPPED_SCENARIOS = Path(__file__).resolve().parents[1] / "fixtures" / "scenarios"
SIMPLE_TRACE = {"expected": "Paris", "output": "Paris"}


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_a_folder_with_only_a_trace_is_a_scenario_with_defaults(tmp_path: Path) -> None:
    _write(tmp_path / "capital_question" / "trace.json", SIMPLE_TRACE)
    scenarios, problems = discover_scenarios(tmp_path)
    assert problems == []
    [scenario] = scenarios
    assert scenario.id == "capital_question"
    assert scenario.title == "Capital question"
    assert scenario.domain == "other"
    assert scenario.declared_fields is None
    assert scenario.overlay is None
    assert scenario.metric_ids is None
    assert not scenario.has_saved_run


def test_nested_folders_take_their_domain_from_the_top_level_folder(tmp_path: Path) -> None:
    _write(tmp_path / "hiring" / "unconfirmed" / "trace.json", SIMPLE_TRACE)
    [scenario], _problems = discover_scenarios(tmp_path)
    assert scenario.id == "hiring/unconfirmed"
    assert scenario.domain == "hiring"


def test_sidecar_files_are_loaded(tmp_path: Path) -> None:
    folder = tmp_path / "demo"
    _write(folder / "trace.json", {"output": "Paris"})
    _write(folder / "declared.json", {"expected": "Paris"})
    _write(folder / "overlay.json", {"run_stalled": False})
    _write(
        folder / "scenario.json",
        {"title": "Capital", "domain": "geo", "condition": "G1", "story": "s", "description": "d", "metrics": ["f1"]},
    )
    (folder / "image.png").write_bytes(b"\x89PNG")
    scenario = load_scenario(folder)
    assert (scenario.title, scenario.domain, scenario.condition) == ("Capital", "geo", "G1")
    assert (scenario.story, scenario.description) == ("s", "d")
    assert scenario.declared_fields == {"expected": "Paris"}
    assert scenario.overlay == {"run_stalled": False}
    assert scenario.metric_ids == ("f1",)
    assert scenario.image_path == folder / "image.png"


def test_metrics_default_to_what_the_saved_run_scored(tmp_path: Path) -> None:
    folder = tmp_path / "demo"
    _write(folder / "trace.json", SIMPLE_TRACE)
    rows = {
        "task_success": {"metric_id": "task_success", "status": "scored", "raw_score": 1.0, "grade": 4},
        "grounding": {"metric_id": "grounding", "status": "na", "na_reason": "missing_evidence"},
    }
    _write(folder / "saved_run.json", {"record": SIMPLE_TRACE, "rows": rows, "rationale": {"task_success": "ok"}})
    scenario = load_scenario(folder)
    assert scenario.metric_ids == ("task_success",)
    assert scenario.saved_record == SIMPLE_TRACE
    assert scenario.saved_rationale == {"task_success": "ok"}


def test_a_lone_non_sidecar_json_counts_as_the_trace(tmp_path: Path) -> None:
    _write(tmp_path / "demo" / "my_export.json", SIMPLE_TRACE)
    _write(tmp_path / "demo" / "declared.json", {"expected": "Paris"})
    scenario = load_scenario(tmp_path / "demo")
    assert scenario.trace_path.name == "my_export.json"


def test_broken_folders_are_reported_without_hiding_the_others(tmp_path: Path) -> None:
    _write(tmp_path / "good" / "trace.json", SIMPLE_TRACE)
    _write(tmp_path / "two_traces" / "a.json", SIMPLE_TRACE)
    _write(tmp_path / "two_traces" / "b.json", SIMPLE_TRACE)
    _write(tmp_path / "bad_declared" / "trace.json", SIMPLE_TRACE)
    (tmp_path / "bad_declared" / "declared.json").write_text("{not json", encoding="utf-8")
    _write(tmp_path / "list_overlay" / "trace.json", SIMPLE_TRACE)
    _write(tmp_path / "list_overlay" / "overlay.json", [1, 2])
    scenarios, problems = discover_scenarios(tmp_path)
    assert [s.id for s in scenarios] == ["good"]
    assert len(problems) == 3
    assert any("two_traces" in p and "several candidate trace files" in p for p in problems)
    assert any("bad_declared" in p and "not valid JSON" in p for p in problems)
    assert any("list_overlay" in p and "must hold a JSON object" in p for p in problems)


def test_a_missing_root_is_a_problem_not_an_exception(tmp_path: Path) -> None:
    scenarios, problems = discover_scenarios(tmp_path / "nope")
    assert scenarios == []
    assert len(problems) == 1


def test_default_dir_honours_the_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SCENARIOS_DIR_ENV, str(tmp_path))
    assert default_scenarios_dir() == tmp_path
    monkeypatch.delenv(SCENARIOS_DIR_ENV)
    assert default_scenarios_dir() == SHIPPED_SCENARIOS


def test_the_shipped_scenarios_all_load() -> None:
    scenarios, problems = discover_scenarios(SHIPPED_SCENARIOS)
    assert problems == []
    curated = [s for s in scenarios if not s.id.startswith("multidomain/")]
    promoted = [s for s in scenarios if s.id.startswith("multidomain/")]
    assert len(curated) >= 12
    assert len(promoted) >= 20
    assert all(s.has_saved_run and s.metric_ids and s.description and s.story for s in curated)
    assert all(s.story and s.metric_ids for s in promoted)


def test_condition_labels_are_explained_from_workflow_and_qualifiers() -> None:
    explained = condition_definition("Confirm gate, two runs")
    assert explained is not None
    assert explained.startswith(WORKFLOW_DEFINITIONS["Confirm gate"])
    assert explained.endswith(QUALIFIER_DEFINITIONS["two runs"])
    assert condition_definition("Sequential") == WORKFLOW_DEFINITIONS["Sequential"]
    assert condition_definition("Something new") is None


def test_every_shipped_scenario_names_a_defined_workflow() -> None:
    scenarios, _problems = discover_scenarios(SHIPPED_SCENARIOS)
    curated = [s for s in scenarios if not s.id.startswith("multidomain/")]
    for scenario in curated:
        workflow = scenario.condition.split(",")[0].strip()
        assert workflow in WORKFLOW_DEFINITIONS, f"{scenario.id}: {scenario.condition!r}"


def _curated_scenario_ids() -> list[str]:
    scenarios, _ = discover_scenarios(SHIPPED_SCENARIOS)
    return sorted(s.id for s in scenarios if not s.id.startswith("multidomain/") and s.has_saved_run)


@pytest.mark.parametrize("scenario_id", _curated_scenario_ids())
def test_a_shipped_scenario_replays_exactly(scenario_id: str) -> None:
    """Re-ingest reproduces the saved record; re-scoring reproduces the saved rows."""
    scenario = load_scenario(SHIPPED_SCENARIOS / scenario_id, root=SHIPPED_SCENARIOS)
    assert scenario.saved_record is not None
    [result] = ingest_scenario(scenario)
    saved_without_judges = {k: v for k, v in scenario.saved_record.items() if not k.endswith("_judge")}
    assert result.record == saved_without_judges
    assert read_trace_payload(scenario) is not None

    saved_run = json.loads((scenario.folder / "saved_run.json").read_text(encoding="utf-8"))
    rescored = {row["metric_id"]: row for row in score_metrics(scenario.saved_record)}
    for metric_id, saved_row in saved_run["rows"].items():
        assert rescored[metric_id] == saved_row, metric_id

    explanation = explain_hard_stops(
        score_selected(scenario.saved_record, set(scenario.metric_ids or ())), result.record
    )
    if explanation["hard_stops"]:
        assert analysis_matches(scenario.saved_analysis, explanation), "hard stop fires but no matching saved analysis"
        assert scenario.saved_analysis is not None
        span_ids = {span.id for span in result.spans or []}
        assert set(scenario.saved_analysis["responsible_steps"]) <= span_ids
