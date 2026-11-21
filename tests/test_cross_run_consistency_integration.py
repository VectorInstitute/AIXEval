"""Full pipeline: real scenarios, repeated live judge calls, cross_run_consistency.

Marked integration_test: needs a real provider key to actually run.
Skips (not fails) when none is configured, same discipline as
test_judge_integration.py.
"""

import os

import pytest
from dotenv import load_dotenv

from aixeval.bundle import build_run_bundle
from aixeval.cross_run_consistency import ProblemRuns, score_cross_run_consistency
from aixeval.judge.config import build_client_from_env
from aixeval.judge.live import extract_live_grounding_evidence, run_live_judges
from aixeval.scorecard import build_profile, score_metrics
from aixeval.ui.scenarios import default_scenarios_dir, discover_scenarios, ingest_scenario, read_trace_payload


_KEY_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")
_METRIC_ID = "grounding"
_REPEATS_PER_PROBLEM = 2
_SCENARIO_IDS = ("healthcare/lab_order_confirmed", "hiring/rejection_email_confirmed")


@pytest.mark.integration_test
def test_real_scenarios_repeated_live_judge_feeds_cross_run_consistency() -> None:
    """Two real demo scenarios, each judged live twice, fed into the aggregate.

    Not asserting specific variance numbers: a live judge's exact
    verdict can vary by model and run, and that run-to-run variance is
    the very thing under test here. What this proves is that real
    scenario fixtures, real repeated live-judge calls, and real bundles
    compose end to end into score_cross_run_consistency's input shape
    and produce a valid scored (or honestly na) aggregate -- not that
    any particular scenario is or isn't stable on a given day.
    """
    load_dotenv()
    if not any(os.environ.get(var) for var in _KEY_ENV_VARS):
        pytest.skip("No provider API key configured; set one in .env to run this live.")

    scenarios, load_problems = discover_scenarios(default_scenarios_dir())
    assert not load_problems, f"demo scenarios failed to load: {load_problems}"
    by_id = {s.id: s for s in scenarios}
    client = build_client_from_env()

    problem_runs: list[ProblemRuns] = []
    for scenario_id in _SCENARIO_IDS:
        scenario = by_id[scenario_id]
        result = ingest_scenario(scenario)[0]
        assert result.trajectory, f"{scenario_id} has no trajectory; grounding can't score it"

        bundles = []
        for _ in range(_REPEATS_PER_PROBLEM):
            updated, _rationale, errors = run_live_judges(
                result.record,
                trajectory=result.trajectory,
                image_path=None,
                selected_ids={_METRIC_ID},
                client=client,
                evidence=extract_live_grounding_evidence(read_trace_payload(scenario), result.spans),
            )
            assert not errors, f"{scenario_id}: live judge call failed: {errors}"
            scored = score_metrics(updated)
            profile = build_profile(scored)
            bundles.append(build_run_bundle(result, scored, profile, judge=f"{type(client).__name__} {client.model}"))
        problem_runs.append(ProblemRuns(scenario_id, bundles))

    result = score_cross_run_consistency(problem_runs, metric_id=_METRIC_ID)

    assert result["status"] == "scored"
    assert result["problems_scored"] == len(_SCENARIO_IDS)
    assert result["excluded_problems"] == []
    assert 0.0 <= result["raw_score"] <= 1.0
    assert 0 <= result["grade"] <= 4
    assert 0.0 <= result["unstable_share"] <= 1.0
    assert len(result["per_problem"]) == len(_SCENARIO_IDS)
