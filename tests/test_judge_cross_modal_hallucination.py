"""judge_cross_modal_hallucination: description vs image_path modes.

Parse / default max_tokens for description mode live in shared suites.
"""

import pytest

from aixeval.judge.cross_modal_hallucination import judge_cross_modal_hallucination


class _FakeClient:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls: list[dict[str, object]] = []
        self.image_calls: list[dict[str, object]] = []

    @property
    def model(self) -> str:
        return "fake-model"

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens})
        return self._reply

    def complete_with_image(self, prompt: str, image_path: str, *, max_tokens: int = 1024) -> str:
        self.image_calls.append({"prompt": prompt, "image_path": image_path, "max_tokens": max_tokens})
        return self._reply


def test_sends_description_and_output_in_the_prompt() -> None:
    client = _FakeClient("SCORE: 0")
    judge_cross_modal_hallucination(
        "the specific output text", client, image_description="the specific description text"
    )
    prompt = client.calls[0]["prompt"]
    assert "the specific description text" in prompt
    assert "the specific output text" in prompt


def test_image_path_mode_calls_complete_with_image_not_complete() -> None:
    client = _FakeClient("SCORE: 0.3")
    score = judge_cross_modal_hallucination("the specific output text", client, image_path="/data/dog.jpg")
    assert score == 0.3
    assert client.calls == []
    assert len(client.image_calls) == 1
    assert client.image_calls[0]["image_path"] == "/data/dog.jpg"
    assert "the specific output text" in client.image_calls[0]["prompt"]


def test_image_path_mode_max_tokens_is_overridable() -> None:
    client = _FakeClient("SCORE: 0")
    judge_cross_modal_hallucination("o", client, image_path="/data/dog.jpg", max_tokens=64)
    assert client.image_calls[0]["max_tokens"] == 64


def test_neither_image_description_nor_image_path_raises() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        judge_cross_modal_hallucination("o", _FakeClient("SCORE: 0"))


def test_both_image_description_and_image_path_raises() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        judge_cross_modal_hallucination("o", _FakeClient("SCORE: 0"), image_description="d", image_path="/data/dog.jpg")
