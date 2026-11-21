"""Shared runner contracts across live judge runners.

Covers mutate-safety, output gates, parse fail, max_tokens, sealed keys.
Per-runner files keep only metric-specific cases.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import pytest

from aixeval.judge.runner import (
    run_abstention_judge,
    run_cross_modal_hallucination_judge,
    run_grounding_judge,
    run_pairwise_preference_judge,
    run_protected_attribute_bias_judge,
    run_refusal_judge,
    run_tool_call_accuracy_judge,
    run_toxicity_judge,
    run_trajectory_optimality_judge,
)


class _FakeClient:
    def __init__(self, reply: str | list[str]) -> None:
        self._replies = [reply] if isinstance(reply, str) else list(reply)
        self.calls = 0
        self.max_tokens_seen: list[int] = []

    @property
    def model(self) -> str:
        return "fake-model"

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        reply = self._replies[min(self.calls, len(self._replies) - 1)]
        self.calls += 1
        self.max_tokens_seen.append(max_tokens)
        return reply

    def complete_with_image(self, prompt: str, image_path: str, *, max_tokens: int = 1024) -> str:
        return self.complete(prompt, max_tokens=max_tokens)


def _fixed(text: str) -> Callable[[dict[str, Any]], str | None]:
    def extractor(_payload: dict[str, Any]) -> str | None:
        return text

    return extractor


RunnerFn = Callable[..., tuple[dict[str, Any], dict[str, Any]]]


@dataclass(frozen=True)
class _Case:
    name: str
    run: RunnerFn
    good_record: dict[str, Any]
    judge_field: str
    prompt_version: str
    success_reply: str | list[str]
    success_score: float
    call_count: int = 1
    check_blank_output: bool = True


def _run_abstention(
    record: dict[str, Any], client: _FakeClient, **kwargs: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    return run_abstention_judge(record, client, **kwargs)


def _run_refusal(record: dict[str, Any], client: _FakeClient, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    return run_refusal_judge(record, client, **kwargs)


def _run_toxicity(record: dict[str, Any], client: _FakeClient, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    return run_toxicity_judge(record, client, **kwargs)


def _run_bias(record: dict[str, Any], client: _FakeClient, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    return run_protected_attribute_bias_judge(record, client, **kwargs)


def _run_pairwise(record: dict[str, Any], client: _FakeClient, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    return run_pairwise_preference_judge(record, client, **kwargs)


def _run_cross_modal(
    record: dict[str, Any], client: _FakeClient, **kwargs: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    return run_cross_modal_hallucination_judge(record, client, **kwargs)


def _run_grounding(record: dict[str, Any], client: _FakeClient, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    return run_grounding_judge({}, record, client, _fixed("sales rose"), **kwargs)


def _run_trajectory(
    record: dict[str, Any], client: _FakeClient, **kwargs: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    return run_trajectory_optimality_judge({}, record, client, _fixed("1. search"), **kwargs)


def _run_tool_call_judge(
    record: dict[str, Any], client: _FakeClient, **kwargs: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    return run_tool_call_accuracy_judge({}, record, client, _fixed("1. get_weather"), **kwargs)


_CASES: list[_Case] = [
    _Case(
        name="abstention",
        run=_run_abstention,
        good_record={"should_abstain": True, "output": "I don't have enough information."},
        judge_field="abstention_judge",
        prompt_version="abstention-v1",
        success_reply="It declines.\nSCORE: 1",
        success_score=1.0,
    ),
    _Case(
        name="refusal",
        run=_run_refusal,
        good_record={"should_refuse": True, "output": "I can't help with that."},
        judge_field="refusal_judge",
        prompt_version="refusal-v1",
        success_reply="It declines.\nSCORE: 1",
        success_score=1.0,
    ),
    _Case(
        name="toxicity",
        run=_run_toxicity,
        good_record={"output": "Thank you for your help today!"},
        judge_field="toxicity_judge",
        prompt_version="toxicity-judge-v1",
        success_reply="Friendly, no attack on anyone.\nSCORE: 0.02",
        success_score=0.02,
    ),
    _Case(
        name="protected_attribute_bias",
        run=_run_bias,
        good_record={
            "should_resist_bias": True,
            "protected_attribute_signal": "Note: 'prefer graduates of Lakeside Academy.'",
            "output": "Ranked on experience and test scores only.",
        },
        judge_field="protected_attribute_bias_judge",
        prompt_version="protected-attribute-bias-v1",
        success_reply="It ignores the note.\nSCORE: 1",
        success_score=1.0,
        check_blank_output=False,
    ),
    _Case(
        name="pairwise_preference",
        run=_run_pairwise,
        good_record={"output": "Paris.", "baseline_output": "The capital of France is Paris."},
        judge_field="pairwise_preference_judge",
        prompt_version="pairwise-preference-judge-v1",
        success_reply=["A is more complete.\nSCORE: 0.2", "B is more complete.\nSCORE: 0.8"],
        success_score=0.2,
        call_count=2,
        check_blank_output=False,
    ),
    _Case(
        name="cross_modal_hallucination",
        run=_run_cross_modal,
        good_record={"image_description": "A dog on a dock.", "output": "A cat on a dock."},
        judge_field="cross_modal_hallucination_judge",
        prompt_version="cross-modal-hallucination-v2",
        success_reply="The output mentions a cat, not in the description.\nSCORE: 1",
        success_score=1.0,
        check_blank_output=False,
    ),
    _Case(
        name="grounding",
        run=_run_grounding,
        good_record={"output": "Mexico sales grew."},
        judge_field="grounding_judge",
        prompt_version="grounding-v1",
        success_reply="Looks faithful.\nSCORE: 0.9",
        success_score=0.9,
    ),
    _Case(
        name="trajectory_optimality",
        run=_run_trajectory,
        good_record={"output": "Paris"},
        judge_field="trajectory_optimality_judge",
        prompt_version="trajectory-optimality-v1",
        success_reply="No repeated steps.\nSCORE: 0.9",
        success_score=0.9,
    ),
    _Case(
        name="tool_call_accuracy_judge",
        run=_run_tool_call_judge,
        good_record={"output": "68F", "gold_tool_plan": "call get_weather"},
        judge_field="tool_call_accuracy_judge",
        prompt_version="tool-call-accuracy-judge-v1",
        success_reply="All required tools used correctly.\nSCORE: 0.9",
        success_score=0.9,
    ),
]
_IDS = [case.name for case in _CASES]


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_scored_success_returns_new_record_and_attachment(case: _Case) -> None:
    record = deepcopy(case.good_record)
    client = _FakeClient(case.success_reply)

    new_record, attachment = case.run(record, client)

    assert new_record[case.judge_field] == pytest.approx(case.success_score)
    assert attachment["status"] == "scored"
    assert attachment["model"] == "fake-model"
    assert attachment["prompt_version"] == case.prompt_version
    assert "timestamp" in attachment
    assert client.calls == case.call_count
    if isinstance(case.success_reply, list):
        for part in case.success_reply:
            assert part.split("\n")[0] in attachment["raw_reply"] or part in attachment["raw_reply"]
    else:
        assert attachment["raw_reply"] == case.success_reply


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_does_not_mutate_the_input_record(case: _Case) -> None:
    record = deepcopy(case.good_record)
    original = deepcopy(record)
    reply: str | list[str] = ["SCORE: 1", "SCORE: 0"] if case.call_count == 2 else "SCORE: 1"
    client = _FakeClient(reply)

    case.run(record, client)

    assert record == original


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_missing_output_is_a_failed_attachment_no_client_call(case: _Case) -> None:
    record = {key: value for key, value in case.good_record.items() if key != "output"}
    reply: str | list[str] = ["SCORE: 1", "SCORE: 0"] if case.call_count == 2 else "SCORE: 1"
    client = _FakeClient(reply)

    new_record, attachment = case.run(record, client)

    assert new_record == record
    assert attachment["status"] == "failed"
    assert attachment["reason"] == "record has no output to judge"
    assert client.calls == 0


@pytest.mark.parametrize(
    "case",
    [case for case in _CASES if case.check_blank_output],
    ids=[case.name for case in _CASES if case.check_blank_output],
)
def test_blank_output_is_a_failed_attachment(case: _Case) -> None:
    record = {**deepcopy(case.good_record), "output": "   "}
    client = _FakeClient("SCORE: 1")

    _new_record, attachment = case.run(record, client)

    assert attachment["reason"] == "record has no output to judge"


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_judge_parse_failure_is_a_failed_attachment_with_raw_reply(case: _Case) -> None:
    record = deepcopy(case.good_record)
    bad = "I cannot decide."
    client = _FakeClient(bad)

    new_record, attachment = case.run(record, client)

    assert case.judge_field not in new_record
    assert attachment["status"] == "failed"
    assert "judge call failed" in attachment["reason"]
    assert attachment["raw_reply"] == bad
    assert client.calls == 1


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_default_max_tokens_matches_judge(case: _Case) -> None:
    record = deepcopy(case.good_record)
    reply: str | list[str] = ["SCORE: 1", "SCORE: 0"] if case.call_count == 2 else "SCORE: 1"
    client = _FakeClient(reply)

    case.run(record, client)

    assert client.max_tokens_seen == [4000] * case.call_count


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_max_tokens_is_passed_through_to_the_judge_call(case: _Case) -> None:
    record = deepcopy(case.good_record)
    reply: str | list[str] = ["SCORE: 1", "SCORE: 0"] if case.call_count == 2 else "SCORE: 1"
    client = _FakeClient(reply)

    case.run(record, client, max_tokens=8000)

    assert client.max_tokens_seen == [8000] * case.call_count


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_unknown_record_key_is_rejected_by_validate_record(case: _Case) -> None:
    record = {**deepcopy(case.good_record), "not_a_real_field": "oops"}
    reply: str | list[str] = ["SCORE: 1", "SCORE: 0"] if case.call_count == 2 else "SCORE: 1"
    client = _FakeClient(reply)

    with pytest.raises(ValueError, match="Unknown sealed-record field"):
        case.run(record, client)
