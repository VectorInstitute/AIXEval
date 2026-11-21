"""suggest_relevant_metrics: prompt a client, parse strict RELEVANT: lines."""

import pytest

from aixeval.judge.metric_suggester import suggest_relevant_metrics


class _FakeClient:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls: list[dict[str, object]] = []

    @property
    def model(self) -> str:
        return "fake-model"

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens})
        return self._reply


def test_parses_one_relevant_line() -> None:
    client = _FakeClient("RELEVANT: task_success - the run has a gold answer")
    result = suggest_relevant_metrics("a QA run", client, metric_ids=("task_success", "clip_score"))
    assert result == {"task_success": "the run has a gold answer"}


def test_parses_several_relevant_lines() -> None:
    reply = "RELEVANT: task_success - has gold\nRELEVANT: grounding - has evidence"
    client = _FakeClient(reply)
    result = suggest_relevant_metrics("a QA run", client, metric_ids=("task_success", "grounding", "clip_score"))
    assert result == {"task_success": "has gold", "grounding": "has evidence"}


def test_reasoning_text_around_relevant_lines_is_ignored() -> None:
    reply = "Let me think.\nRELEVANT: task_success - has gold\nThat's my pick."
    client = _FakeClient(reply)
    result = suggest_relevant_metrics("a QA run", client, metric_ids=("task_success",))
    assert result == {"task_success": "has gold"}


def test_em_dash_separator_also_parses() -> None:
    client = _FakeClient("RELEVANT: task_success — has gold")
    result = suggest_relevant_metrics("a QA run", client, metric_ids=("task_success",))
    assert result == {"task_success": "has gold"}


def test_unknown_metric_id_in_reply_is_dropped() -> None:
    reply = "RELEVANT: task_success - has gold\nRELEVANT: not_a_real_metric - made up"
    client = _FakeClient(reply)
    result = suggest_relevant_metrics("a QA run", client, metric_ids=("task_success",))
    assert result == {"task_success": "has gold"}


def test_no_relevant_lines_raises() -> None:
    client = _FakeClient("I don't think any metrics apply here.")
    with pytest.raises(ValueError, match="No parseable RELEVANT"):
        suggest_relevant_metrics("a QA run", client, metric_ids=("task_success",))


def test_sends_description_and_catalog_in_the_prompt() -> None:
    client = _FakeClient("RELEVANT: task_success - has gold")
    suggest_relevant_metrics("a very specific run description", client, metric_ids=("task_success",))
    prompt = client.calls[0]["prompt"]
    assert "a very specific run description" in prompt
    assert "task_success:" in prompt


def test_default_metric_ids_is_the_full_catalog() -> None:
    client = _FakeClient("RELEVANT: task_success - has gold")
    suggest_relevant_metrics("a run", client)
    prompt = client.calls[0]["prompt"]
    assert "clip_score:" in prompt  # opt-in metric, proves ALL_METRIC_IDS is the default


def test_max_tokens_is_overridable() -> None:
    client = _FakeClient("RELEVANT: task_success - has gold")
    suggest_relevant_metrics("a run", client, metric_ids=("task_success",), max_tokens=64)
    assert client.calls[0]["max_tokens"] == 64


def test_no_facts_given_means_no_facts_block_in_the_prompt() -> None:
    client = _FakeClient("RELEVANT: task_success - has gold")
    suggest_relevant_metrics("a run", client, metric_ids=("task_success",))
    assert "RUN FACTS" not in client.calls[0]["prompt"]


def test_facts_appear_in_the_prompt_when_given() -> None:
    client = _FakeClient("RELEVANT: task_success - has gold")
    suggest_relevant_metrics(
        "a run",
        client,
        metric_ids=("task_success",),
        source_kind="trace",
        has_trajectory=True,
        num_distinct_agents=2,
        has_image=False,
    )
    prompt = client.calls[0]["prompt"]
    assert "RUN FACTS" in prompt
    assert "source kind: trace" in prompt
    assert "has a step-by-step trajectory: yes" in prompt
    assert "distinct agents observed: 2 (multi-agent)" in prompt
    assert "reference image attached: no" in prompt


def test_single_agent_count_is_labeled_single_agent_not_multi() -> None:
    client = _FakeClient("RELEVANT: task_success - has gold")
    suggest_relevant_metrics("a run", client, metric_ids=("task_success",), num_distinct_agents=1)
    assert "1 (single-agent)" in client.calls[0]["prompt"]


def test_only_the_facts_given_are_rendered_not_every_field() -> None:
    client = _FakeClient("RELEVANT: task_success - has gold")
    suggest_relevant_metrics("a run", client, metric_ids=("task_success",), has_image=True)
    prompt = client.calls[0]["prompt"]
    assert "reference image attached: yes" in prompt
    assert "trajectory" not in prompt
    assert "agents observed" not in prompt
