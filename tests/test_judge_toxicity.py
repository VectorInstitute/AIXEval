"""judge_toxicity: prompt shape. Parse / max_tokens live in shared suites."""

from aixeval.judge.toxicity import judge_toxicity


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


def test_sends_output_in_the_prompt() -> None:
    client = _FakeClient("SCORE: 1")
    judge_toxicity("the specific output text", client)
    prompt = client.calls[0]["prompt"]
    assert "the specific output text" in prompt
