"""run_trajectory_optimality_judge: trajectory extraction cases.

Shared output/mutate/max_tokens/sealed-key cases live in test_judge_runner_wiring.
"""

from aixeval.judge.runner import run_trajectory_optimality_judge


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
    record = {"output": "Paris"}
    client = _FakeClient("SCORE: 1")

    new_record, attachment = run_trajectory_optimality_judge({}, record, client, _no_trajectory)

    assert "trajectory_optimality_judge" not in new_record
    assert new_record == record
    assert attachment["status"] == "failed"
    assert attachment["reason"] == "no trajectory extracted from the dump"
    assert "trajectory_excerpt" not in attachment
    assert "raw_reply" not in attachment
    assert client.calls == 0


def test_extract_trajectory_receives_the_raw_payload() -> None:
    seen = []

    def extractor(payload: dict) -> str | None:
        seen.append(payload)
        return "1. search"

    record = {"output": "Paris"}
    client = _FakeClient("SCORE: 1")
    payload = {"some": "dump"}

    run_trajectory_optimality_judge(payload, record, client, extractor)

    assert seen == [payload]
