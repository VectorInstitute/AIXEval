"""judge_trajectory_optimality: prompt shape.

Parse / max_tokens live in shared suites.
"""

from aixeval.judge.trajectory_optimality import judge_trajectory_optimality


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


def test_sends_trajectory_and_output_in_the_prompt() -> None:
    client = _FakeClient("SCORE: 1")
    judge_trajectory_optimality("the specific trajectory text", "the specific output text", client)
    prompt = client.calls[0]["prompt"]
    assert "the specific trajectory text" in prompt
    assert "the specific output text" in prompt
