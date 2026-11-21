"""aixeval score / report / reband / replay: the thin CLI, via main()."""

import inspect
import json
from pathlib import Path

import pytest

import aixeval.judge.config as config_module
from aixeval import cli
from aixeval.cli import main


FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_score_prints_a_card(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["score", str(FIXTURES / "langfuse_gist.json")])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "capability:" in out
    assert "hard stops:" in out


def test_score_writes_a_bundle(tmp_path: Path) -> None:
    out_path = tmp_path / "run.json"
    exit_code = main(["score", str(FIXTURES / "langfuse_gist.json"), "--out", str(out_path)])
    assert exit_code == 0
    assert out_path.exists()
    bundle = json.loads(out_path.read_text(encoding="utf-8"))
    assert bundle["schema_version"] == "1"
    assert "profile" in bundle
    assert "scored_results" in bundle


def test_score_creates_missing_out_parent_directories(tmp_path: Path) -> None:
    out_path = tmp_path / "runs" / "nested" / "run.json"
    exit_code = main(["score", str(FIXTURES / "langfuse_gist.json"), "--out", str(out_path)])
    assert exit_code == 0
    assert out_path.is_file()


def test_score_all_na_card_explains_missing_evidence(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dump = tmp_path / "empty.json"
    dump.write_text(json.dumps({"output": "hello"}), encoding="utf-8")
    exit_code = main(["score", str(dump)])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "every pillar is N/A" in out


def test_score_auto_loads_scenario_sidecars(tmp_path: Path) -> None:
    dump = FIXTURES / "scenarios" / "hiring" / "resume_injection_followed" / "trace.json"
    out_path = tmp_path / "run.json"
    exit_code = main(["score", str(dump), "--out", str(out_path)])
    assert exit_code == 0
    bundle = json.loads(out_path.read_text(encoding="utf-8"))
    assert bundle["record"]["mas_unsafe_steps"] == 3
    ids = {row["metric_id"] for row in bundle["scored_results"]}
    assert "task_success" in ids
    assert "ocr_text_read_accuracy" not in ids
    assert "wer" not in ids


def test_score_metrics_flag_keeps_only_named_ids(tmp_path: Path) -> None:
    out_path = tmp_path / "run.json"
    exit_code = main(
        ["score", str(FIXTURES / "langfuse_gist.json"), "--metrics", "task_success", "--out", str(out_path)]
    )
    assert exit_code == 0
    bundle = json.loads(out_path.read_text(encoding="utf-8"))
    assert {row["metric_id"] for row in bundle["scored_results"]} == {"task_success"}


def test_score_unknown_metrics_is_a_handled_error(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["score", str(FIXTURES / "langfuse_gist.json"), "--metrics", "not_a_real_metric"])
    err = capsys.readouterr().err
    assert exit_code == 1
    assert "Unknown metric id" in err


def test_score_capped_card_prints_uncapped_grades(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    declared = tmp_path / "declared.json"
    declared.write_text(json.dumps({"high_impact_in_scope": True, "unsafe_taken": True}), encoding="utf-8")
    exit_code = main(["score", str(FIXTURES / "langfuse_gist.json"), "--declared-fields", str(declared)])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "unsafe_action_cap" in out
    assert "capability: 0/4" in out
    assert "uncapped (before hard-stop caps):" in out
    assert "capability 4/4" in out
    assert "every filled question capped at 0" in out
    assert "unsafe_taken=True" in out


def test_score_applies_declared_fields(tmp_path: Path) -> None:
    declared_path = tmp_path / "declared.json"
    declared_path.write_text(json.dumps({"should_refuse": False}), encoding="utf-8")
    out_path = tmp_path / "run.json"
    exit_code = main(
        ["score", str(FIXTURES / "langfuse_gist.json"), "--declared-fields", str(declared_path), "--out", str(out_path)]
    )
    assert exit_code == 0
    bundle = json.loads(out_path.read_text(encoding="utf-8"))
    assert bundle["record"]["should_refuse"] is False


def test_score_missing_file_is_a_handled_error(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["score", "/does/not/exist.json"])
    err = capsys.readouterr().err
    assert exit_code == 1
    assert "error:" in err


def test_report_round_trips_a_bundle(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_path = tmp_path / "run.json"
    main(["score", str(FIXTURES / "langfuse_gist.json"), "--out", str(out_path)])
    capsys.readouterr()  # discard score's own output

    exit_code = main(["report", str(out_path)])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "source: trace" in out
    assert "per-metric detail:" in out
    assert "task_success:" in out


def test_score_audit_contamination_flag_prints_the_leak(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["score", str(FIXTURES / "langfuse_gold_in_input.json"), "--audit-contamination"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "contamination flagged:" in out
    assert "Paris" in out


def test_score_without_audit_does_not_flag_gold_in_input(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["score", str(FIXTURES / "langfuse_gold_in_input.json")])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "contamination flagged:" not in out
    assert "contamination audit:" not in out


def test_score_audit_contamination_prints_when_clean(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    declared = tmp_path / "declared.json"
    declared.write_text(json.dumps({"expected": "completely unrelated gold phrase xyz"}), encoding="utf-8")
    exit_code = main(
        [
            "score",
            str(FIXTURES / "langfuse_gist.json"),
            "--declared-fields",
            str(declared),
            "--audit-contamination",
        ]
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "contamination audit: no leak found" in out
    assert "contamination flagged:" not in out


def test_report_shows_contamination_evidence_when_flagged(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bundle_path = tmp_path / "flagged.json"
    bundle_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "source_kind": "simple_dump",
                "record": {},
                "spans": None,
                "trajectory": None,
                "contamination_evidence": "overlap 72% with declared gold",
                "scored_results": [],
                "profile": dict.fromkeys(
                    (
                        "capability",
                        "robustness",
                        "safety",
                        "fairness",
                        "transparency",
                        "governance",
                        "oversight",
                        "efficiency",
                    )
                )
                | {"applied_hard_stops": []},
            }
        ),
        encoding="utf-8",
    )
    exit_code = main(["report", str(bundle_path)])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "contamination flagged: overlap 72%" in out


def test_report_prints_the_saved_judge(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bundle_path = tmp_path / "judged.json"
    bundle_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "source_kind": "simple_dump",
                "judge": "vllm · Qwen2.5-7B-Instruct",
                "record": {},
                "spans": None,
                "trajectory": None,
                "contamination_evidence": None,
                "scored_results": [],
                "profile": dict.fromkeys(
                    (
                        "capability",
                        "robustness",
                        "safety",
                        "fairness",
                        "transparency",
                        "governance",
                        "oversight",
                        "efficiency",
                    )
                )
                | {"applied_hard_stops": []},
            }
        ),
        encoding="utf-8",
    )
    exit_code = main(["report", str(bundle_path)])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "judge: vllm · Qwen2.5-7B-Instruct" in out


def test_reband_with_no_options_matches_the_shipped_policy(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_path = tmp_path / "run.json"
    main(["score", str(FIXTURES / "langfuse_gist.json"), "--out", str(out_path)])
    capsys.readouterr()

    main(["report", str(out_path)])
    report_out = capsys.readouterr().out

    main(["reband", str(out_path)])
    reband_out = capsys.readouterr().out

    for dimension in ("capability:", "transparency:", "hard stops:"):
        assert dimension in report_out
        assert dimension in reband_out
    assert "per-metric detail:" in reband_out
    assert "verbalized_confidence:" in reband_out


def test_reband_with_stricter_cutoffs_changes_a_grade(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_path = tmp_path / "run.json"
    main(["score", str(FIXTURES / "langfuse_gist.json"), "--out", str(out_path)])
    capsys.readouterr()

    exit_code = main(["reband", str(out_path), "--cutoffs", "0.3,0.5,0.7,0.95"])
    out = capsys.readouterr().out
    assert exit_code == 0
    # verbalized_confidence raw_score 0.900 no longer clears a 0.95 grade-4 cutoff.
    assert "transparency: 3/4" in out


def test_reband_hard_stop_override(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bundle_path = tmp_path / "unsafe.json"
    bundle_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "source_kind": "simple_dump",
                "record": {},
                "spans": None,
                "trajectory": None,
                "contamination_evidence": None,
                "scored_results": [
                    {"metric_id": "task_success", "status": "scored", "raw_score": 1.0, "grade": 4},
                    {"metric_id": "unsafe_action", "status": "scored", "raw_score": 0.18, "grade": 1},
                ],
                "profile": {},
            }
        ),
        encoding="utf-8",
    )
    exit_code = main(["reband", str(bundle_path), "--hard-stop", "unsafe_action_cap=0.15"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "unsafe_action_cap" in out
    assert "capability: 0/4" in out


def test_reband_unknown_hard_stop_name_is_a_handled_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_path = tmp_path / "run.json"
    main(["score", str(FIXTURES / "langfuse_gist.json"), "--out", str(out_path)])
    capsys.readouterr()

    exit_code = main(["reband", str(out_path), "--hard-stop", "not_a_rule=0.5"])
    err = capsys.readouterr().err
    assert exit_code == 1
    assert "Unknown hard-stop rule" in err


def test_no_subcommand_exits_nonzero() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_replay_matches_a_clean_bundle(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_path = tmp_path / "run.json"
    main(["score", str(FIXTURES / "langfuse_gist.json"), "--out", str(out_path)])
    capsys.readouterr()

    exit_code = main(["replay", str(out_path)])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "replay matches exactly" in out


def test_replay_reports_a_tampered_bundle(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_path = tmp_path / "run.json"
    main(["score", str(FIXTURES / "langfuse_gist.json"), "--out", str(out_path)])
    capsys.readouterr()

    bundle = json.loads(out_path.read_text(encoding="utf-8"))
    bundle["profile"]["capability"] = 0
    out_path.write_text(json.dumps(bundle), encoding="utf-8")

    exit_code = main(["replay", str(out_path)])
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "replay MISMATCH" in out
    assert "dimension capability" in out


def test_compare_matches_two_identical_bundles(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    main(["score", str(FIXTURES / "langfuse_gist.json"), "--out", str(path_a)])
    main(["score", str(FIXTURES / "langfuse_gist.json"), "--out", str(path_b)])
    capsys.readouterr()

    exit_code = main(["compare", str(path_a), str(path_b)])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "re-run holds up" in out


def test_compare_flags_divergence_beyond_the_default_tolerance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    main(["score", str(FIXTURES / "langfuse_gist.json"), "--out", str(path_a)])
    capsys.readouterr()

    bundle = json.loads(path_a.read_text(encoding="utf-8"))
    bundle["profile"] = {**bundle["profile"], "capability": 0}  # saved run grades capability 4
    path_b.write_text(json.dumps(bundle), encoding="utf-8")

    exit_code = main(["compare", str(path_a), str(path_b)])
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "DIVERGED" in out
    assert "capability" in out


def test_compare_flags_na_versus_scored(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    main(["score", str(FIXTURES / "langfuse_gist.json"), "--out", str(path_a)])
    capsys.readouterr()

    bundle = json.loads(path_a.read_text(encoding="utf-8"))
    bundle["profile"] = {**bundle["profile"], "robustness": 4}
    path_b.write_text(json.dumps(bundle), encoding="utf-8")

    exit_code = main(["compare", str(path_a), str(path_b)])
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "N/A vs scored" in out
    assert "robustness" in out


def test_compare_max_grade_drift_override_tolerates_a_wider_gap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    main(["score", str(FIXTURES / "langfuse_gist.json"), "--out", str(path_a)])
    capsys.readouterr()

    bundle = json.loads(path_a.read_text(encoding="utf-8"))
    bundle["profile"] = {**bundle["profile"], "capability": 2}  # saved run grades capability 4, drift 2
    path_b.write_text(json.dumps(bundle), encoding="utf-8")

    exit_code = main(["compare", str(path_a), str(path_b), "--max-grade-drift", "2"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "re-run holds up" in out


class _FakeClient:
    """Returns replies from a fixed list, one per call, in order."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self.calls = 0

    @property
    def model(self) -> str:
        return "fake-model"

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        reply = self._replies[self.calls]
        self.calls += 1
        return reply


def test_test_retest_metric_attaches_rerun_scores_and_scores_the_metric(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_client = _FakeClient(["SCORE: 0.8", "SCORE: 0.85", "SCORE: 0.82"])
    monkeypatch.setattr(config_module, "build_client_from_env", lambda: fake_client)

    out_path = tmp_path / "run.json"
    exit_code = main(
        [
            "score",
            str(FIXTURES / "langfuse_gist.json"),
            "--test-retest-metric",
            "abstention",
            "--test-retest-n",
            "3",
            "--out",
            str(out_path),
        ]
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    assert fake_client.calls == 3
    assert "test-retest: abstention scored [0.8, 0.85, 0.82]" in out

    bundle = json.loads(out_path.read_text(encoding="utf-8"))
    assert bundle["record"]["rerun_scores"] == [0.8, 0.85, 0.82]
    assert bundle["record"]["abstention_judge"] == 0.82  # the last call's actual verdict, not discarded
    by_id = {row["metric_id"]: row for row in bundle["scored_results"]}
    assert by_id["test_retest_stability"]["status"] == "scored"


def test_test_retest_n_one_warns_and_leaves_rerun_scores_unset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_client = _FakeClient(["SCORE: 0.8"])
    monkeypatch.setattr(config_module, "build_client_from_env", lambda: fake_client)

    out_path = tmp_path / "run.json"
    exit_code = main(
        [
            "score",
            str(FIXTURES / "langfuse_gist.json"),
            "--test-retest-metric",
            "abstention",
            "--test-retest-n",
            "1",
            "--out",
            str(out_path),
        ]
    )
    err = capsys.readouterr().err
    assert exit_code == 0
    assert "only 1 successful abstention rerun(s)" in err

    bundle = json.loads(out_path.read_text(encoding="utf-8"))
    assert "rerun_scores" not in bundle["record"]
    assert bundle["record"]["abstention_judge"] == 0.8  # the one successful call's verdict, still kept
    by_id = {row["metric_id"]: row for row in bundle["scored_results"]}
    assert by_id["test_retest_stability"]["status"] == "na"


def test_cli_live_judges_import_from_judge_not_ui() -> None:
    source = inspect.getsource(cli._apply_live_judges) + inspect.getsource(cli._apply_test_retest)
    assert "aixeval.ui" not in source
    assert "aixeval.judge.live" in source
    assert "extract_live_grounding_evidence" in source


def test_test_retest_unknown_metric_is_a_handled_error(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["score", str(FIXTURES / "langfuse_gist.json"), "--test-retest-metric", "not_a_real_metric"])
    err = capsys.readouterr().err
    assert exit_code == 1
    assert "Unknown live-judge metric" in err


def _write_dump(path: Path, *, expected: str, output: str) -> None:
    path.write_text(json.dumps({"expected": expected, "output": output}), encoding="utf-8")


def test_cross_run_consistency_flags_the_unstable_problem(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    suite = tmp_path / "suite"
    stable, flaky = suite / "stable", suite / "flaky"
    for folder, outputs in ((stable, ("Paris", "Paris")), (flaky, ("Madrid", "Barcelona"))):
        for i, output in enumerate(outputs):
            dump = tmp_path / f"{folder.name}_{i}.json"
            _write_dump(dump, expected="Paris" if folder is stable else "Madrid", output=output)
            main(["score", str(dump), "--out", str(folder / f"run{i}.json")])
    capsys.readouterr()

    exit_code = main(["cross-run-consistency", str(suite), "--metric", "task_success", "--seed", "0"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "task_success: grade" in out
    assert "problems_scored: 2" in out
    assert "not fully stable: flaky" in out


def test_cross_run_consistency_excludes_a_problem_with_only_one_bundle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = tmp_path / "suite"
    ok, lonely = suite / "ok", suite / "lonely"
    for i in range(2):
        dump = tmp_path / f"ok_{i}.json"
        _write_dump(dump, expected="Paris", output="Paris")
        main(["score", str(dump), "--out", str(ok / f"run{i}.json")])
    dump = tmp_path / "lonely_0.json"
    _write_dump(dump, expected="Paris", output="Paris")
    main(["score", str(dump), "--out", str(lonely / "run0.json")])
    capsys.readouterr()

    exit_code = main(["cross-run-consistency", str(suite), "--metric", "task_success"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "problems_scored: 1" in out
    assert "excluded (too few usable runs)" in out
    assert "lonely: 1 usable run(s)" in out


def test_cross_run_consistency_no_problem_has_enough_runs_is_a_handled_outcome(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = tmp_path / "suite"
    dump = tmp_path / "only.json"
    _write_dump(dump, expected="Paris", output="Paris")
    main(["score", str(dump), "--out", str(suite / "lonely" / "run0.json")])
    capsys.readouterr()

    exit_code = main(["cross-run-consistency", str(suite), "--metric", "task_success"])
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "could not score" in out


def test_cross_run_consistency_missing_directory_is_a_handled_error(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["cross-run-consistency", "/no/such/dir", "--metric", "task_success"])
    err = capsys.readouterr().err
    assert exit_code == 1
    assert "is not a directory" in err


def test_cross_run_consistency_seed_is_reproducible(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    suite = tmp_path / "suite"
    for folder, outputs in ((suite / "a", ("Paris", "Rome")), (suite / "b", ("Paris", "Paris"))):
        for i, output in enumerate(outputs):
            dump = tmp_path / f"{folder.name}_{i}.json"
            _write_dump(dump, expected="Paris", output=output)
            main(["score", str(dump), "--out", str(folder / f"run{i}.json")])
    capsys.readouterr()

    main(["cross-run-consistency", str(suite), "--metric", "task_success", "--seed", "7"])
    out_a = capsys.readouterr().out
    main(["cross-run-consistency", str(suite), "--metric", "task_success", "--seed", "7"])
    out_b = capsys.readouterr().out
    assert out_a == out_b
