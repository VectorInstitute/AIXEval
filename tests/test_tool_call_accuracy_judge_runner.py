"""run_tool_call_accuracy_judge: gold plan / trajectory extraction cases.

Shared output/mutate/max_tokens/sealed-key cases live in test_judge_runner_wiring.
"""

from aixeval.judge.runner import run_tool_call_accuracy_judge


class _FakeClient:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls = 0

    @property
    def model(self) -> str:
        return "fake-model"

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        self.calls += 1
        return self._reply


def _no_trajectory(_payload: dict) -> str | None:
    return None


def _fixed_trajectory(text: str):
    def extractor(payload: dict) -> str | None:
        return text

    return extractor


def test_no_trajectory_is_a_failed_attachment_no_client_call() -> None:
    record = {"output": "68F", "gold_tool_plan": "call get_weather"}
    client = _FakeClient("SCORE: 1")

    new_record, attachment = run_tool_call_accuracy_judge({}, record, client, _no_trajectory)

    assert "tool_call_accuracy_judge" not in new_record
    assert new_record == record
    assert attachment["status"] == "failed"
    assert attachment["reason"] == "no trajectory extracted from the dump"
    assert "trajectory_excerpt" not in attachment
    assert "raw_reply" not in attachment
    assert client.calls == 0


def test_missing_gold_tool_plan_is_a_failed_attachment_no_client_call() -> None:
    record = {"output": "68F"}
    client = _FakeClient("SCORE: 1")

    new_record, attachment = run_tool_call_accuracy_judge({}, record, client, _fixed_trajectory("1. get_weather"))

    assert new_record == record
    assert attachment["status"] == "failed"
    assert attachment["reason"] == "record has no gold_tool_plan to judge against"
    assert client.calls == 0


def test_blank_gold_tool_plan_is_a_failed_attachment() -> None:
    record = {"output": "68F", "gold_tool_plan": "   "}
    client = _FakeClient("SCORE: 1")

    _new_record, attachment = run_tool_call_accuracy_judge({}, record, client, _fixed_trajectory("1. get_weather"))

    assert attachment["reason"] == "record has no gold_tool_plan to judge against"


def test_extract_trajectory_receives_the_raw_payload() -> None:
    seen = []

    def extractor(payload: dict) -> str | None:
        seen.append(payload)
        return "1. get_weather"

    record = {"output": "68F", "gold_tool_plan": "call get_weather"}
    client = _FakeClient("SCORE: 1")
    payload = {"some": "dump"}

    run_tool_call_accuracy_judge(payload, record, client, extractor)

    assert seen == [payload]
