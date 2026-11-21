"""run_pairwise_preference_judge: baseline/reference-specific cases.

Shared output/mutate/max_tokens/sealed-key cases live in test_judge_runner_wiring.
"""

from aixeval.judge.runner import run_pairwise_preference_judge


class _FakeClient:
    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self.calls = 0
        self.prompts: list[str] = []

    @property
    def model(self) -> str:
        return "fake-model"

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        reply = self._replies[self.calls]
        self.calls += 1
        self.prompts.append(prompt)
        return reply


def test_missing_baseline_is_a_failed_attachment_no_client_call() -> None:
    record = {"output": "Paris."}
    client = _FakeClient(["SCORE: 1", "SCORE: 0"])

    new_record, attachment = run_pairwise_preference_judge(record, client)

    assert new_record == record
    assert attachment["status"] == "failed"
    assert attachment["reason"] == "record has no baseline_output to compare against"
    assert client.calls == 0


def test_blank_baseline_is_a_failed_attachment() -> None:
    record = {"output": "Paris.", "baseline_output": "   "}
    client = _FakeClient(["SCORE: 1", "SCORE: 0"])

    _new_record, attachment = run_pairwise_preference_judge(record, client)

    assert attachment["reason"] == "record has no baseline_output to compare against"


def test_reference_passed_through_when_expected_is_present() -> None:
    record = {"output": "Paris.", "baseline_output": "Paris, France.", "expected": "the gold answer"}
    client = _FakeClient(["SCORE: 1", "SCORE: 0"])

    run_pairwise_preference_judge(record, client)

    assert "the gold answer" in client.prompts[0]
    assert "the gold answer" in client.prompts[1]


def test_no_reference_when_expected_is_absent() -> None:
    record = {"output": "Paris.", "baseline_output": "Paris, France."}
    client = _FakeClient(["SCORE: 1", "SCORE: 0"])

    run_pairwise_preference_judge(record, client)

    assert "REFERENCE ANSWER" not in client.prompts[0]
