"""build_run_bundle / write_run_bundle / read_run_bundle: one portable run export."""

import json
from pathlib import Path

import pytest

from aixeval.bundle import (
    BUNDLE_SCHEMA_VERSION,
    build_run_bundle,
    compare_bundles,
    read_run_bundle,
    replay_bundle,
    write_run_bundle,
)
from aixeval.ingest.pipeline import ingest_any
from aixeval.scorecard.metrics import score_metrics
from aixeval.scorecard.profile import build_profile


def _scored_simple_dump():
    result = ingest_any({"expected": "Paris", "output": "Paris"})
    scored = score_metrics(result.record)
    profile = build_profile(scored)
    return result, scored, profile


def test_bundle_carries_schema_version() -> None:
    result, scored, profile = _scored_simple_dump()
    bundle = build_run_bundle(result, scored, profile)
    assert bundle["schema_version"] == BUNDLE_SCHEMA_VERSION


def test_bundle_carries_record_scored_results_and_profile() -> None:
    result, scored, profile = _scored_simple_dump()
    bundle = build_run_bundle(result, scored, profile)
    assert bundle["record"] == result.record
    assert bundle["scored_results"] == scored
    assert bundle["profile"] == profile
    assert bundle["source_kind"] == "simple_dump"


def test_bundle_records_the_judge_only_when_given() -> None:
    result, scored, profile = _scored_simple_dump()
    assert "judge" not in build_run_bundle(result, scored, profile)
    bundle = build_run_bundle(result, scored, profile, judge="vllm · gemma-3-12b-it")
    assert bundle["judge"] == "vllm · gemma-3-12b-it"
    assert replay_bundle(bundle)["matches"]


def test_bundle_spans_is_none_for_a_simple_dump() -> None:
    result, scored, profile = _scored_simple_dump()
    bundle = build_run_bundle(result, scored, profile)
    assert bundle["spans"] is None


def test_bundle_flattens_spans_for_a_trace() -> None:
    payload = {
        "traj": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    }
    result = ingest_any(payload)
    scored = score_metrics(result.record)
    profile = build_profile(scored)
    bundle = build_run_bundle(result, scored, profile)
    assert bundle["spans"] is not None
    assert all(isinstance(span, dict) for span in bundle["spans"])
    assert {"id", "kind", "agent", "input", "output"} <= bundle["spans"][0].keys()


def test_bundle_is_json_serializable() -> None:
    result, scored, profile = _scored_simple_dump()
    bundle = build_run_bundle(result, scored, profile)
    json.dumps(bundle)  # must not raise


def test_write_creates_missing_parent_directories(tmp_path) -> None:
    result, scored, profile = _scored_simple_dump()
    bundle = build_run_bundle(result, scored, profile)
    path = tmp_path / "runs" / "nested" / "run.json"
    write_run_bundle(bundle, path)
    assert path.is_file()


def test_write_then_read_round_trips(tmp_path) -> None:
    result, scored, profile = _scored_simple_dump()
    bundle = build_run_bundle(result, scored, profile)
    path = tmp_path / "run.json"

    write_run_bundle(bundle, path)
    read_back = read_run_bundle(path)

    # JSON has no tuple type, so applied_hard_stops round-trips as a list,
    # not a tuple; compare it separately rather than with a strict ==.
    assert list(read_back["profile"].pop("applied_hard_stops")) == list(bundle["profile"].pop("applied_hard_stops"))
    assert read_back == bundle


def test_read_rejects_a_missing_schema_version(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"record": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        read_run_bundle(path)


def test_read_rejects_ui_saved_run_with_a_clearer_error() -> None:
    path = Path(__file__).parent.parent / "fixtures/scenarios/legal/case_holding_sequential/saved_run.json"
    with pytest.raises(ValueError, match="saved_run.json"):
        read_run_bundle(path)


def test_read_rejects_an_unknown_schema_version(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": "99", "record": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="99"):
        read_run_bundle(path)


def test_write_formats_as_readable_indented_json(tmp_path) -> None:
    result, scored, profile = _scored_simple_dump()
    bundle = build_run_bundle(result, scored, profile)
    path = tmp_path / "run.json"
    write_run_bundle(bundle, path)
    text = path.read_text(encoding="utf-8")
    assert "\n" in text
    assert text.startswith("{\n")


def test_replay_matches_a_selected_metric_subset() -> None:
    result, scored, _profile = _scored_simple_dump()
    subset = [row for row in scored if row["metric_id"] == "task_success"]
    bundle = build_run_bundle(result, subset, build_profile(subset))
    replay = replay_bundle(bundle)
    assert replay["matches"] is True
    assert replay["metric_diffs"] == []
    assert replay["profile_diffs"] == []


def test_replay_matches_a_clean_bundle() -> None:
    result, scored, profile = _scored_simple_dump()
    bundle = build_run_bundle(result, scored, profile)
    replay = replay_bundle(bundle)
    assert replay["matches"] is True
    assert replay["metric_diffs"] == []
    assert replay["profile_diffs"] == []


def test_replay_matches_after_a_real_json_round_trip(tmp_path) -> None:
    result, scored, profile = _scored_simple_dump()
    bundle = build_run_bundle(result, scored, profile)
    path = tmp_path / "run.json"
    write_run_bundle(bundle, path)
    replay = replay_bundle(read_run_bundle(path))
    assert replay["matches"] is True


def test_replay_needs_no_model_call_for_an_already_written_judge_verdict() -> None:
    # grounding_judge is already on the record, as if a live judge ran earlier --
    # score_metrics must read it back, not call a judge again, for replay to work.
    result = _scored_simple_dump()[0]
    record = {**result.record, "grounding_judge": 0.85}
    scored = score_metrics(record)
    profile = build_profile(scored)
    from dataclasses import replace  # noqa: PLC0415

    bundle = build_run_bundle(replace(result, record=record), scored, profile)
    replay = replay_bundle(bundle)
    assert replay["matches"] is True


def test_replay_needs_no_raw_payload_for_an_already_written_tool_call_verdict() -> None:
    # tool_call_accuracy_judge is already on the record; replay must not need
    # the original raw payload extract_tool_calls_* would otherwise require.
    result = _scored_simple_dump()[0]
    record = {**result.record, "gold_tool_calls": [{"tool": "x", "args": {}}], "tool_call_accuracy_judge": 0.5}
    scored = score_metrics(record)
    profile = build_profile(scored)
    from dataclasses import replace  # noqa: PLC0415

    bundle = build_run_bundle(replace(result, record=record), scored, profile)
    replay = replay_bundle(bundle)
    assert replay["matches"] is True


def test_replay_reports_a_tampered_metric_result() -> None:
    result, scored, profile = _scored_simple_dump()
    bundle = build_run_bundle(result, scored, profile)
    bundle["scored_results"] = [
        {**row, "raw_score": 0.0, "grade": 0} if row["metric_id"] == "task_success" else row
        for row in bundle["scored_results"]
    ]
    replay = replay_bundle(bundle)
    assert replay["matches"] is False
    diff = next(d for d in replay["metric_diffs"] if d["metric_id"] == "task_success")
    assert diff["saved"]["raw_score"] == 0.0
    assert diff["replayed"]["raw_score"] == 1.0


def test_replay_reports_a_tampered_profile_dimension() -> None:
    result, scored, profile = _scored_simple_dump()
    bundle = build_run_bundle(result, scored, profile)
    bundle["profile"] = {**bundle["profile"], "capability": 0}
    replay = replay_bundle(bundle)
    assert replay["matches"] is False
    diff = next(d for d in replay["profile_diffs"] if d["dimension"] == "capability")
    assert diff["saved"] == 0
    assert diff["replayed"] == 4


def test_replay_ignores_the_tuple_vs_list_difference_on_applied_hard_stops() -> None:
    result, scored, profile = _scored_simple_dump()
    bundle = build_run_bundle(result, scored, profile)
    bundle["profile"]["applied_hard_stops"] = list(bundle["profile"]["applied_hard_stops"])
    replay = replay_bundle(bundle)
    assert replay["matches"] is True


def test_compare_bundles_identical_profiles_do_not_diverge() -> None:
    result, scored, profile = _scored_simple_dump()
    bundle = build_run_bundle(result, scored, profile)
    comparison = compare_bundles(bundle, bundle)
    assert comparison == {"diverged": False, "dimension_diffs": []}


def test_compare_bundles_flags_drift_beyond_the_default_tolerance() -> None:
    result, scored, profile = _scored_simple_dump()
    bundle_a = build_run_bundle(result, scored, profile)
    bundle_b = build_run_bundle(result, scored, {**profile, "capability": 2})  # saved capability is 4
    comparison = compare_bundles(bundle_a, bundle_b)
    assert comparison["diverged"] is True
    diff = next(d for d in comparison["dimension_diffs"] if d["dimension"] == "capability")
    assert diff == {"dimension": "capability", "grade_a": 4, "grade_b": 2, "drift": 2}


def test_compare_bundles_tolerates_drift_within_max_grade_drift() -> None:
    result, scored, profile = _scored_simple_dump()
    bundle_a = build_run_bundle(result, scored, profile)
    bundle_b = build_run_bundle(result, scored, {**profile, "capability": 3})  # saved capability is 4, drift 1
    comparison = compare_bundles(bundle_a, bundle_b, max_grade_drift=1)
    assert comparison == {"diverged": False, "dimension_diffs": []}


def test_compare_bundles_max_grade_drift_zero_requires_an_exact_match() -> None:
    result, scored, profile = _scored_simple_dump()
    bundle_a = build_run_bundle(result, scored, profile)
    bundle_b = build_run_bundle(result, scored, {**profile, "capability": 3})
    comparison = compare_bundles(bundle_a, bundle_b, max_grade_drift=0)
    assert comparison["diverged"] is True


def test_compare_bundles_ignores_applied_hard_stops() -> None:
    result, scored, profile = _scored_simple_dump()
    bundle_a = build_run_bundle(result, scored, {**profile, "applied_hard_stops": ["unsafe_action_cap"]})
    bundle_b = build_run_bundle(result, scored, {**profile, "applied_hard_stops": []})
    comparison = compare_bundles(bundle_a, bundle_b)
    assert comparison == {"diverged": False, "dimension_diffs": []}


def test_compare_bundles_skips_a_dimension_that_is_na_on_both_sides() -> None:
    result, scored, profile = _scored_simple_dump()
    bundle_a = build_run_bundle(result, scored, {**profile, "robustness": None})
    bundle_b = build_run_bundle(result, scored, {**profile, "robustness": None})
    comparison = compare_bundles(bundle_a, bundle_b)
    assert comparison == {"diverged": False, "dimension_diffs": []}


def test_compare_bundles_flags_a_dimension_that_is_na_on_only_one_side() -> None:
    result, scored, profile = _scored_simple_dump()
    bundle_a = build_run_bundle(result, scored, {**profile, "robustness": None})
    bundle_b = build_run_bundle(result, scored, {**profile, "robustness": 4})
    comparison = compare_bundles(bundle_a, bundle_b)
    assert comparison["diverged"] is True
    diff = next(d for d in comparison["dimension_diffs"] if d["dimension"] == "robustness")
    assert diff == {"dimension": "robustness", "grade_a": None, "grade_b": 4, "drift": None}
