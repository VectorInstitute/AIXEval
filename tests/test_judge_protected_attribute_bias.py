"""judge_protected_attribute_bias + detector.

Parse / max_tokens for SCORE live in shared suites.
"""

import pytest

from aixeval.judge.protected_attribute_bias import (
    judge_detect_protected_attribute_signal,
    judge_protected_attribute_bias,
)


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


def test_sends_signal_and_output_in_the_prompt() -> None:
    client = _FakeClient("SCORE: 1")
    judge_protected_attribute_bias("the specific signal text", "the specific output text", client)
    prompt = client.calls[0]["prompt"]
    assert "the specific signal text" in prompt
    assert "the specific output text" in prompt


# judge_detect_protected_attribute_signal: a different question (does a
# signal exist at all), its own SIGNAL: line, its own tests below.


def test_detector_finds_a_quoted_signal() -> None:
    reply = "There's a note about school preference.\nSIGNAL: prefer graduates of Lakeside Academy"
    client = _FakeClient(reply)
    assert judge_detect_protected_attribute_signal("text", client) == "prefer graduates of Lakeside Academy"


def test_detector_strips_surrounding_quotes() -> None:
    client = _FakeClient('SIGNAL: "prefer graduates of Lakeside Academy"')
    assert judge_detect_protected_attribute_signal("text", client) == "prefer graduates of Lakeside Academy"


def test_detector_none_is_no_candidate() -> None:
    client = _FakeClient("Nothing like that here.\nSIGNAL: NONE")
    assert judge_detect_protected_attribute_signal("text", client) is None


def test_detector_none_is_case_insensitive() -> None:
    client = _FakeClient("SIGNAL: none")
    assert judge_detect_protected_attribute_signal("text", client) is None


def test_detector_takes_the_last_signal_line_if_several() -> None:
    reply = "Draft: SIGNAL: maybe this\nOn reflection:\nSIGNAL: NONE"
    client = _FakeClient(reply)
    assert judge_detect_protected_attribute_signal("text", client) is None


def test_detector_no_signal_line_raises() -> None:
    client = _FakeClient("I looked but didn't conclude.")
    with pytest.raises(ValueError, match="No parseable SIGNAL"):
        judge_detect_protected_attribute_signal("text", client)


def test_detector_sends_input_text_in_the_prompt() -> None:
    client = _FakeClient("SIGNAL: NONE")
    judge_detect_protected_attribute_signal("the specific input text", client)
    prompt = client.calls[0]["prompt"]
    assert "the specific input text" in prompt


def test_detector_default_max_tokens_has_headroom_for_reasoning_models() -> None:
    client = _FakeClient("SIGNAL: NONE")
    judge_detect_protected_attribute_signal("text", client)
    assert client.calls[0]["max_tokens"] == 4000


def test_detector_max_tokens_is_overridable() -> None:
    client = _FakeClient("SIGNAL: NONE")
    judge_detect_protected_attribute_signal("text", client, max_tokens=64)
    assert client.calls[0]["max_tokens"] == 64
