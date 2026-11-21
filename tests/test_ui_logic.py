"""ui.logic plus judge.live: run_live_judges / score_selected / explain_row."""

import pytest

from aixeval.ingest.pipeline import IngestResult
from aixeval.ingest.trace import Span
from aixeval.ingest.trace_taubench import extract_tool_calls_taubench
from aixeval.judge.evidence import extract_tool_calls_langfuse
from aixeval.judge.live import extract_live_grounding_evidence, live_judge_is_possible, run_live_judges
from aixeval.ui.logic import (
    SAVED_VERDICTS_LABEL,
    compare_rows,
    derive_run_facts,
    describe_judges,
    explain_row,
    judge_label,
    merge_bias_signal_into_declared,
    replays_judge_verdicts,
    run_deterministic_tool_call_check,
    score_all,
    score_selected,
    span_rows,
    tool_call_extractor_for,
)


class _FakeClient:
    def __init__(self, reply: str = "SCORE: 0.7", image_reply: str | None = None, raises: bool = False) -> None:
        self._reply = reply
        self._image_reply = image_reply if image_reply is not None else reply
        self._raises = raises
        self.calls: list[str] = []

    @property
    def model(self) -> str:
        return "fake-model"

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        self.calls.append("complete")
        if self._raises:
            return "no score line here"
        return self._reply

    def complete_with_image(self, prompt: str, image_path: str, *, max_tokens: int = 1024) -> str:
        self.calls.append("complete_with_image")
        if self._raises:
            return "no score line here"
        return self._image_reply


def test_runs_a_trajectory_judge_when_trajectory_and_output_present() -> None:
    client = _FakeClient("Looks fine.\nSCORE: 0.8")
    record = {"output": "the final answer"}
    updated, rationale, errors = run_live_judges(
        record, trajectory="1. step one", image_path=None, selected_ids={"trajectory_optimality"}, client=client
    )
    assert updated["trajectory_optimality_judge"] == 0.8
    assert "Looks fine." in rationale["trajectory_optimality"]
    assert errors == {}


def test_skips_trajectory_metric_when_no_trajectory_available() -> None:
    client = _FakeClient("SCORE: 0.8")
    record = {"output": "the final answer"}
    updated, rationale, errors = run_live_judges(
        record, trajectory=None, image_path=None, selected_ids={"trajectory_optimality", "grounding"}, client=client
    )
    assert "trajectory_optimality_judge" not in updated
    assert "grounding_judge" not in updated
    assert rationale == {}
    assert errors == {}
    assert client.calls == []


def test_grounding_skipped_when_only_trajectory_is_present() -> None:
    client = _FakeClient("SCORE: 0.6")
    record = {"output": "Paris"}
    updated, rationale, errors = run_live_judges(
        record, trajectory="1. step one -- output: planning", image_path=None, selected_ids={"grounding"}, client=client
    )
    assert "grounding_judge" not in updated
    assert rationale == {}
    assert errors == {}
    assert client.calls == []


def test_grounding_uses_evidence_not_trajectory() -> None:
    class _RecordingClient(_FakeClient):
        def __init__(self) -> None:
            super().__init__("SCORE: 0.6")
            self.prompts: list[str] = []

        def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
            self.prompts.append(prompt)
            return super().complete(prompt, max_tokens=max_tokens)

    client = _RecordingClient()
    record = {"output": "Paris"}
    updated, _rationale, errors = run_live_judges(
        record,
        trajectory="1. [SPAN] agent -- input: Capital? -- output: planning",
        image_path=None,
        selected_ids={"grounding"},
        client=client,
        evidence="axis ticks: 1QFY22",
    )
    assert updated["grounding_judge"] == 0.6
    assert errors == {}
    assert updated["judge_provenance"]["grounding"] == {"model": "fake-model", "prompt_version": "grounding-v1"}
    assert "axis ticks: 1QFY22" in client.prompts[0]
    assert "1. [SPAN]" not in client.prompts[0]


def test_grounding_runs_without_a_trajectory_when_evidence_is_present() -> None:
    client = _FakeClient("SCORE: 0.6")
    record = {"output": "Paris"}
    updated, _rationale, errors = run_live_judges(
        record,
        trajectory=None,
        image_path=None,
        selected_ids={"grounding"},
        client=client,
        evidence="France's capital is Paris.",
    )
    assert updated["grounding_judge"] == 0.6
    assert errors == {}


def test_toxicity_runs_without_a_trajectory() -> None:
    client = _FakeClient("SCORE: 0.1")
    record = {"output": "a perfectly polite answer"}
    updated, rationale, errors = run_live_judges(
        record, trajectory=None, image_path=None, selected_ids={"toxicity"}, client=client
    )
    assert updated["toxicity_judge"] == 0.1
    assert errors == {}
    assert updated["judge_provenance"]["toxicity"] == {"model": "fake-model", "prompt_version": "toxicity-judge-v1"}


def test_image_metric_skipped_without_an_image_path() -> None:
    client = _FakeClient("SCORE: 0.0")
    record = {"output": "a description"}
    updated, _rationale, errors = run_live_judges(
        record, trajectory=None, image_path=None, selected_ids={"cross_modal_hallucination"}, client=client
    )
    assert "cross_modal_hallucination_judge" not in updated
    assert errors == {}
    assert client.calls == []


def test_image_metric_runs_when_image_path_given() -> None:
    client = _FakeClient("SCORE: 0.0")
    record = {"output": "a faithful description"}
    updated, _rationale, errors = run_live_judges(
        record, trajectory=None, image_path="/tmp/fake.jpg", selected_ids={"cross_modal_hallucination"}, client=client
    )
    assert updated["cross_modal_hallucination_judge"] == 0.0
    assert client.calls == ["complete_with_image"]
    assert errors == {}


def test_abstention_runs_without_a_trajectory() -> None:
    client = _FakeClient("SCORE: 1.0")
    record = {"output": "I'm not sure, I don't have enough information."}
    updated, rationale, errors = run_live_judges(
        record, trajectory=None, image_path=None, selected_ids={"abstention"}, client=client
    )
    assert updated["abstention_judge"] == 1.0
    assert rationale["abstention"]
    assert errors == {}


def test_refusal_runs_without_a_trajectory() -> None:
    client = _FakeClient("SCORE: 1.0")
    record = {"output": "I can't help with that."}
    updated, rationale, errors = run_live_judges(
        record, trajectory=None, image_path=None, selected_ids={"refusal"}, client=client
    )
    assert updated["refusal_judge"] == 1.0
    assert rationale["refusal"]
    assert errors == {}


def test_tool_call_accuracy_skipped_without_a_trajectory() -> None:
    client = _FakeClient("SCORE: 0.9")
    record = {"output": "the final answer", "gold_tool_plan": "call search then calculator"}
    updated, _rationale, errors = run_live_judges(
        record, trajectory=None, image_path=None, selected_ids={"tool_call_accuracy"}, client=client
    )
    assert "tool_call_accuracy_judge" not in updated
    assert errors == {}
    assert client.calls == []


def test_tool_call_accuracy_skipped_without_a_declared_gold_plan() -> None:
    client = _FakeClient("SCORE: 0.9")
    record = {"output": "the final answer"}
    updated, _rationale, errors = run_live_judges(
        record, trajectory="1. called search", image_path=None, selected_ids={"tool_call_accuracy"}, client=client
    )
    assert "tool_call_accuracy_judge" not in updated
    assert errors == {}
    assert client.calls == []


def test_tool_call_accuracy_runs_with_trajectory_and_gold_plan() -> None:
    client = _FakeClient("SCORE: 0.9")
    record = {"output": "the final answer", "gold_tool_plan": "call search then calculator"}
    updated, rationale, errors = run_live_judges(
        record,
        trajectory="1. called search",
        image_path=None,
        selected_ids={"tool_call_accuracy"},
        client=client,
    )
    assert updated["tool_call_accuracy_judge"] == 0.9
    assert rationale["tool_call_accuracy"]
    assert errors == {}


def test_pairwise_preference_skipped_without_a_declared_baseline() -> None:
    client = _FakeClient("SCORE: 1.0")
    record = {"output": "the final answer"}
    updated, _rationale, errors = run_live_judges(
        record, trajectory=None, image_path=None, selected_ids={"pairwise_preference"}, client=client
    )
    assert "pairwise_preference_judge" not in updated
    assert errors == {}
    assert client.calls == []


def test_pairwise_preference_runs_with_a_declared_baseline() -> None:
    # Position-bias correction makes two calls per verdict, output shown in
    # each position once; the fake client returns the same "A wins" reply
    # both times, so output wins its position-A call and loses its
    # position-B call, averaging to 0.5, not 1.0. Proves both calls
    # actually ran, not just that a score came back.
    client = _FakeClient("SCORE: 1.0")
    record = {"output": "the final answer", "baseline_output": "a worse answer"}
    updated, rationale, errors = run_live_judges(
        record, trajectory=None, image_path=None, selected_ids={"pairwise_preference"}, client=client
    )
    assert updated["pairwise_preference_judge"] == 0.5
    assert rationale["pairwise_preference"]
    assert errors == {}
    assert client.calls == ["complete", "complete"]


def test_protected_attribute_bias_skipped_without_a_declared_signal() -> None:
    client = _FakeClient("SCORE: 1.0")
    record = {"output": "Hired."}
    updated, _rationale, errors = run_live_judges(
        record, trajectory=None, image_path=None, selected_ids={"protected_attribute_bias"}, client=client
    )
    assert "protected_attribute_bias_judge" not in updated
    assert errors == {}
    assert client.calls == []


def test_protected_attribute_bias_runs_with_a_declared_signal() -> None:
    client = _FakeClient("SCORE: 1.0")
    record = {
        "output": "Ranked on experience only.",
        "protected_attribute_signal": "Note: 'prefer graduates of Lakeside Academy.'",
    }
    updated, rationale, errors = run_live_judges(
        record, trajectory=None, image_path=None, selected_ids={"protected_attribute_bias"}, client=client
    )
    assert updated["protected_attribute_bias_judge"] == 1.0
    assert rationale["protected_attribute_bias"]
    assert errors == {}


def test_a_bad_judge_reply_is_caught_per_metric_not_fatal() -> None:
    client = _FakeClient(raises=True)
    record = {"output": "the final answer"}
    updated, rationale, errors = run_live_judges(
        record, trajectory="1. step one", image_path=None, selected_ids={"trajectory_optimality"}, client=client
    )
    assert "trajectory_optimality_judge" not in updated
    assert rationale == {}
    assert "trajectory_optimality" in errors


def test_skips_when_record_has_no_usable_output() -> None:
    client = _FakeClient("SCORE: 0.8")
    updated, _rationale, errors = run_live_judges(
        {}, trajectory="1. step one", image_path=None, selected_ids={"trajectory_optimality"}, client=client
    )
    assert "trajectory_optimality_judge" not in updated
    assert errors == {}
    assert client.calls == []


def test_non_live_metric_ids_are_simply_ignored() -> None:
    client = _FakeClient("SCORE: 0.8")
    record = {"output": "answer", "expected": "answer"}
    updated, _rationale, _errors = run_live_judges(
        record, trajectory=None, image_path=None, selected_ids={"task_success", "bleu"}, client=client
    )
    assert updated == record
    assert client.calls == []


def test_score_selected_filters_to_the_chosen_ids() -> None:
    record = {"expected": "yes", "output": "yes"}
    rows = score_selected(record, {"task_success"})
    assert [row["metric_id"] for row in rows] == ["task_success"]
    assert rows[0]["status"] == "scored"


def test_score_selected_includes_toxicity_only_when_selected() -> None:
    record = {"output": "fine", "toxicity_judge": 0.0}
    rows = score_selected(record, {"toxicity"})
    assert [row["metric_id"] for row in rows] == ["toxicity"]
    assert rows[0]["status"] == "scored"

    rows_unselected = score_selected(record, set())
    assert rows_unselected == []


def test_explain_row_for_a_scored_metric_with_rationale() -> None:
    row = {"metric_id": "trajectory_optimality", "status": "scored", "raw_score": 0.7, "grade": 3}
    text = explain_row(row, {"trajectory_optimality": "Steps looked mostly direct."})
    assert "Did the steps taken visibly waste effort" in text
    assert "Raw score 0.700 -> grade 3/4" in text
    assert "Steps looked mostly direct." in text


def test_explain_row_for_a_scored_metric_without_rationale() -> None:
    row = {"metric_id": "task_success", "status": "scored", "raw_score": 1.0, "grade": 4}
    text = explain_row(row, {})
    assert "Raw score 1.000 -> grade 4/4" in text
    assert "Judge's reasoning" not in text


def test_explain_row_for_an_na_metric() -> None:
    row = {"metric_id": "grounding", "status": "na", "na_reason": "missing_evidence"}
    text = explain_row(row, {})
    assert "N/A:" in text
    assert "lacks fields this metric needs" in text


def test_derive_run_facts_for_a_simple_dump_has_no_trajectory_or_agents() -> None:
    result = IngestResult(record={"output": "yes"}, spans=None, trajectory=None, source_kind="simple_dump")
    facts = derive_run_facts(result, image_path=None)
    assert facts == {
        "source_kind": "simple_dump",
        "has_trajectory": False,
        "num_distinct_agents": None,
        "has_image": False,
    }


def test_derive_run_facts_counts_distinct_agents_from_real_spans() -> None:
    spans = [
        Span(id="a1", kind="agent", agent="planner"),
        Span(id="a2", kind="agent", agent="worker", parent_id="a1"),
    ]
    result = IngestResult(record={"output": "yes"}, spans=spans, trajectory="1. ...", source_kind="trace")
    facts = derive_run_facts(result, image_path=None)
    assert facts["source_kind"] == "trace"
    assert facts["has_trajectory"] is True
    assert facts["num_distinct_agents"] == 2


def test_derive_run_facts_single_agent_trace_counts_one() -> None:
    spans = [Span(id="a1", kind="agent", agent="solo"), Span(id="t1", kind="tool", parent_id="a1")]
    result = IngestResult(record={}, spans=spans, trajectory="1. ...", source_kind="trace")
    facts = derive_run_facts(result, image_path=None)
    assert facts["num_distinct_agents"] == 1


def test_derive_run_facts_image_path_sets_has_image() -> None:
    result = IngestResult(record={}, spans=None, trajectory=None, source_kind="simple_dump")
    assert derive_run_facts(result, image_path="/tmp/pic.jpg")["has_image"] is True
    assert derive_run_facts(result, image_path=None)["has_image"] is False


def test_live_judge_is_possible_never_touches_a_real_provider() -> None:
    # No client is passed in at all; if this somehow tried to make a
    # real call, there would be no Provider to call, and it would crash
    # loudly rather than silently succeed with a fabricated answer.
    record = {"output": "the final answer"}
    assert live_judge_is_possible("toxicity", record, trajectory=None, image_path=None) is True


def test_live_judge_is_possible_false_without_output() -> None:
    assert live_judge_is_possible("toxicity", {}, trajectory=None, image_path=None) is False


def test_live_judge_is_possible_false_for_an_unknown_metric_id() -> None:
    record = {"output": "the final answer"}
    assert live_judge_is_possible("task_success", record, trajectory=None, image_path=None) is False


def test_live_judge_is_possible_matches_trajectory_optimality_gate() -> None:
    record = {"output": "the final answer"}
    assert live_judge_is_possible("trajectory_optimality", record, trajectory=None, image_path=None) is False
    assert live_judge_is_possible("trajectory_optimality", record, trajectory="1. step", image_path=None) is True


def test_live_judge_is_possible_matches_grounding_gate() -> None:
    record = {"output": "the final answer"}
    assert live_judge_is_possible("grounding", record, trajectory="1. step", image_path=None) is False
    assert live_judge_is_possible("grounding", record, trajectory=None, image_path=None, evidence="source text") is True


def test_extract_live_grounding_evidence_uses_langfuse_extractor() -> None:
    payload = {"input": "root question", "observations": [{"input": "tool result"}]}
    assert extract_live_grounding_evidence(payload) == "root question\n\ntool result"


def test_extract_live_grounding_evidence_falls_back_to_span_inputs() -> None:
    spans = [Span(id="s1", kind="llm", input="what the agent saw", output="Paris")]
    evidence = extract_live_grounding_evidence(None, spans)
    assert evidence == "what the agent saw"
    assert evidence is not None and "Paris" not in evidence


def test_extract_live_grounding_evidence_prefers_langfuse_dump_over_spans() -> None:
    payload = {"input": "root question", "observations": [{"input": "obs"}]}
    spans = [Span(id="s1", kind="llm", input="obs", output="should not appear")]
    evidence = extract_live_grounding_evidence(payload, spans)
    assert evidence == "root question\n\nobs"
    assert evidence is not None and "should not appear" not in evidence


def test_extract_live_grounding_evidence_none_without_inputs() -> None:
    assert extract_live_grounding_evidence(None, None) is None
    assert extract_live_grounding_evidence({"observations": []}, None) is None


def test_live_judge_is_possible_matches_cross_modal_hallucination_gate() -> None:
    record = {"output": "a description"}
    assert live_judge_is_possible("cross_modal_hallucination", record, trajectory=None, image_path=None) is False
    assert live_judge_is_possible("cross_modal_hallucination", record, trajectory=None, image_path="/tmp/x.jpg") is True


def test_live_judge_is_possible_matches_tool_call_accuracy_gate() -> None:
    with_plan = {"output": "the final answer", "gold_tool_plan": "use search then calculator"}
    without_plan = {"output": "the final answer"}
    assert live_judge_is_possible("tool_call_accuracy", with_plan, trajectory="1. step", image_path=None) is True
    assert live_judge_is_possible("tool_call_accuracy", without_plan, trajectory="1. step", image_path=None) is False
    assert live_judge_is_possible("tool_call_accuracy", with_plan, trajectory=None, image_path=None) is False


def test_live_judge_is_possible_matches_pairwise_preference_gate() -> None:
    with_baseline = {"output": "the final answer", "baseline_output": "a worse answer"}
    without_baseline = {"output": "the final answer"}
    assert live_judge_is_possible("pairwise_preference", with_baseline, trajectory=None, image_path=None) is True
    assert live_judge_is_possible("pairwise_preference", without_baseline, trajectory=None, image_path=None) is False


def test_live_judge_is_possible_matches_protected_attribute_bias_gate() -> None:
    with_signal = {"output": "Ranked on experience only.", "protected_attribute_signal": "prefers school X"}
    without_signal = {"output": "Ranked on experience only."}
    assert live_judge_is_possible("protected_attribute_bias", with_signal, trajectory=None, image_path=None) is True
    assert live_judge_is_possible("protected_attribute_bias", without_signal, trajectory=None, image_path=None) is False


def test_merge_bias_signal_sets_both_fields() -> None:
    merged = merge_bias_signal_into_declared(None, "prefers school X")
    assert merged == {"should_resist_bias": True, "protected_attribute_signal": "prefers school X"}


def test_merge_bias_signal_keeps_existing_declared_fields() -> None:
    merged = merge_bias_signal_into_declared({"expected": "yes"}, "prefers school X")
    assert merged == {
        "expected": "yes",
        "should_resist_bias": True,
        "protected_attribute_signal": "prefers school X",
    }


def test_merge_bias_signal_overwrites_a_prior_signal() -> None:
    merged = merge_bias_signal_into_declared(
        {"should_resist_bias": False, "protected_attribute_signal": "old"}, "new signal"
    )
    assert merged["should_resist_bias"] is True
    assert merged["protected_attribute_signal"] == "new signal"


def test_merge_bias_signal_does_not_mutate_the_input_dict() -> None:
    original = {"expected": "yes"}
    merge_bias_signal_into_declared(original, "prefers school X")
    assert original == {"expected": "yes"}


def test_tool_call_extractor_for_picks_langfuse_for_observations_shape() -> None:
    assert tool_call_extractor_for({"observations": []}) is extract_tool_calls_langfuse


def test_tool_call_extractor_for_picks_taubench_for_traj_shape() -> None:
    assert tool_call_extractor_for({"traj": []}) is extract_tool_calls_taubench


def test_tool_call_extractor_for_returns_none_for_an_unsupported_shape() -> None:
    assert tool_call_extractor_for({"resourceSpans": []}) is None
    assert tool_call_extractor_for({"spans": []}) is None


def test_run_deterministic_tool_call_check_scores_a_taubench_payload() -> None:
    payload = {
        "traj": [
            {"role": "user", "content": "cancel order #100"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "cancel_pending_order", "arguments": '{"order_id": "#100"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"status": "cancelled"}'},
        ]
    }
    record = {"output": "done", "gold_tool_calls": [{"tool": "cancel_pending_order", "args": {"order_id": "#100"}}]}
    updated, evidence = run_deterministic_tool_call_check(payload, record)
    assert updated["tool_call_accuracy_judge"] == 1.0
    assert evidence is None  # perfect match, nothing to explain


def test_run_deterministic_tool_call_check_surfaces_a_mismatch_as_evidence() -> None:
    payload = {
        "traj": [
            {"role": "user", "content": "cancel order #100"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "function": {"name": "send_email", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ]
    }
    record = {"output": "done", "gold_tool_calls": [{"tool": "cancel_pending_order", "args": {"order_id": "#100"}}]}
    updated, evidence = run_deterministic_tool_call_check(payload, record)
    assert updated["tool_call_accuracy_judge"] == 0.0
    assert evidence is not None
    assert "missing gold call(s): cancel_pending_order" in evidence
    assert "extra/unexpected call(s): send_email" in evidence


def test_run_deterministic_tool_call_check_unchanged_without_gold_tool_calls() -> None:
    payload = {"traj": [{"role": "user", "content": "hi"}]}
    record = {"output": "done"}
    updated, evidence = run_deterministic_tool_call_check(payload, record)
    assert updated == record
    assert evidence is None


def test_run_deterministic_tool_call_check_unchanged_for_unsupported_shape() -> None:
    payload = {"resourceSpans": []}
    record = {"output": "done", "gold_tool_calls": [{"tool": "x", "args": {}}]}
    updated, evidence = run_deterministic_tool_call_check(payload, record)
    assert updated == record
    assert evidence is None


class _RateLimitError(Exception):
    """Stand-in for a provider SDK error such as ``openai.RateLimitError``."""


class _FailsOnPromptClient(_FakeClient):
    def __init__(self, fail_marker: str) -> None:
        super().__init__("SCORE: 0.9")
        self._fail_marker = fail_marker

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        if self._fail_marker in prompt.lower():
            msg = "429 Too Many Requests"
            raise _RateLimitError(msg)
        return super().complete(prompt, max_tokens=max_tokens)


def test_a_provider_sdk_error_is_isolated_to_its_own_metric() -> None:
    client = _FailsOnPromptClient("toxic")
    record = {"output": "the final answer"}
    updated, _rationale, errors = run_live_judges(
        record,
        trajectory="1. step one",
        image_path=None,
        selected_ids={"toxicity", "grounding"},
        client=client,
        evidence="source text the answer should stay faithful to",
    )
    assert set(errors) == {"toxicity"}
    assert "_RateLimitError" in errors["toxicity"]
    assert "429" in errors["toxicity"]
    assert "toxicity_judge" not in updated
    assert updated["grounding_judge"] == 0.9


def test_a_verdict_already_on_the_record_is_not_re_judged() -> None:
    client = _FakeClient("SCORE: 0.1")
    record = {"output": "the final answer", "toxicity_judge": 0.7}
    updated, rationale, errors = run_live_judges(
        record, trajectory=None, image_path=None, selected_ids={"toxicity"}, client=client
    )
    assert updated["toxicity_judge"] == 0.7
    assert rationale == {}
    assert errors == {}
    assert client.calls == []


def test_score_all_reports_toxicity_na_without_a_verdict_and_loads_no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def _must_not_run(_text: str) -> float:
        msg = "the local toxicity classifier must never load from the UI"
        raise AssertionError(msg)

    monkeypatch.setattr("aixeval.scorecard.toxicity._default_classifier", _must_not_run)
    rows = {row["metric_id"]: row for row in score_all({"output": "a polite answer"})}
    assert rows["toxicity"] == {"metric_id": "toxicity", "status": "na", "na_reason": "missing_evidence"}


def test_score_all_scores_toxicity_from_a_saved_verdict() -> None:
    rows = {row["metric_id"]: row for row in score_all({"output": "fine", "toxicity_judge": 0.0})}
    assert rows["toxicity"]["status"] == "scored"
    assert rows["toxicity"]["grade"] == 4


def test_span_rows_orders_parents_before_children_and_keeps_sibling_order() -> None:
    spans = [
        Span(id="t2", kind="tool", parent_id="a1", source_type="second_tool"),
        Span(id="t1", kind="tool", parent_id="a1", source_type="first_listed_tool"),
        Span(id="a1", kind="agent", agent="planner"),
        Span(id="l1", kind="llm", parent_id="t1"),
    ]
    rows = span_rows(spans)
    assert [(row["id"], row["depth"]) for row in rows] == [("a1", 0), ("t2", 1), ("t1", 1), ("l1", 2)]


def test_span_rows_shows_an_orphan_as_a_root_and_computes_duration() -> None:
    spans = [
        Span(id="x", kind="tool", parent_id="missing", start="2026-10-27T16:30:00Z", end="2026-10-27T16:30:02.5Z"),
    ]
    rows = span_rows(spans)
    assert rows[0]["depth"] == 0
    assert rows[0]["duration_s"] == 2.5


def test_span_rows_duration_is_none_for_missing_or_bad_times() -> None:
    rows = span_rows(
        [Span(id="a", kind="agent", start="not a time", end="2026-10-27T16:30:00Z"), Span(id="b", kind="agent")]
    )
    assert [row["duration_s"] for row in rows] == [None, None]


def test_compare_rows_lines_up_grades_and_drops_metrics_na_in_both() -> None:
    rows_a = [
        {"metric_id": "task_success", "status": "scored", "raw_score": 1.0, "grade": 4},
        {"metric_id": "unsafe_action", "status": "scored", "raw_score": 0.0, "grade": 4},
        {"metric_id": "grounding", "status": "na", "na_reason": "missing_evidence"},
    ]
    rows_b = [
        {"metric_id": "task_success", "status": "scored", "raw_score": 1.0, "grade": 4},
        {"metric_id": "unsafe_action", "status": "scored", "raw_score": 1.0, "grade": 0},
        {"metric_id": "grounding", "status": "na", "na_reason": "missing_evidence"},
        {"metric_id": "refusal", "status": "scored", "raw_score": 1.0, "grade": 4},
    ]
    assert compare_rows(rows_a, rows_b) == [
        {"metric_id": "refusal", "grade_a": None, "grade_b": 4, "delta": None},
        {"metric_id": "task_success", "grade_a": 4, "grade_b": 4, "delta": 0},
        {"metric_id": "unsafe_action", "grade_a": 4, "grade_b": 0, "delta": -4},
    ]


def test_judge_label_names_provider_and_model() -> None:
    assert judge_label("vllm", "gemma-3-12b-it") == "vllm · gemma-3-12b-it"


def test_describe_judges_covers_live_replayed_both_and_neither() -> None:
    live = judge_label("vllm", "gemma-3-12b-it")
    assert describe_judges(live_judge=live, replayed=False) == live
    assert describe_judges(live_judge=None, replayed=True) == SAVED_VERDICTS_LABEL
    assert describe_judges(live_judge=live, replayed=True) == f"{live} + {SAVED_VERDICTS_LABEL}"
    assert describe_judges(live_judge=None, replayed=False) is None


def test_describe_judges_uses_saved_provenance_when_present() -> None:
    replay_record = {
        "grounding_judge": 1.0,
        "judge_provenance": {"grounding": {"model": "claude-opus-5-5", "prompt_version": "grounding-v1"}},
    }
    assert describe_judges(live_judge=None, replayed=True, replay_record=replay_record) == "claude-opus-5-5"


def test_replays_judge_verdicts_only_for_a_selected_judge_field_on_the_saved_record() -> None:
    saved = {"output": "x", "grounding_judge": 1.0}
    assert replays_judge_verdicts(saved, {"grounding", "task_success"})
    assert not replays_judge_verdicts(saved, {"task_success"})
    assert not replays_judge_verdicts(saved, {"refusal"})
    assert not replays_judge_verdicts(None, {"grounding"})
