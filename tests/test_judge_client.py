"""ClaudeClient.complete: one prompt in, first text block out. No live API here."""

from pathlib import Path

import pytest

from aixeval.judge.client import ClaudeClient


class _Block:
    def __init__(self, block_type: str, text: str = "") -> None:
        self.type = block_type
        self.text = text


class _Response:
    def __init__(self, content: list[_Block], stop_reason: str = "end_turn") -> None:
        self.content = content
        self.stop_reason = stop_reason


class _FakeMessages:
    def __init__(self, response: _Response) -> None:
        self._response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _Response:
        self.calls.append(kwargs)
        return self._response


class _FakeAnthropic:
    def __init__(self, response: _Response) -> None:
        self.messages = _FakeMessages(response)


def test_returns_first_text_block() -> None:
    fake = _FakeAnthropic(_Response([_Block("text", "0.8")]))
    client = ClaudeClient(client=fake)
    assert client.complete("judge this") == "0.8"


def test_skips_non_text_blocks_before_the_text_one() -> None:
    fake = _FakeAnthropic(_Response([_Block("thinking", "..."), _Block("text", "yes")]))
    client = ClaudeClient(client=fake)
    assert client.complete("judge this") == "yes"


def test_no_text_block_raises() -> None:
    fake = _FakeAnthropic(_Response([_Block("thinking", "...")], stop_reason="refusal"))
    client = ClaudeClient(client=fake)
    with pytest.raises(ValueError, match="no text block"):
        client.complete("judge this")


def test_sends_model_and_prompt() -> None:
    fake = _FakeAnthropic(_Response([_Block("text", "ok")]))
    client = ClaudeClient(model="claude-sonnet-5", client=fake)
    client.complete("say ok", max_tokens=16)
    assert fake.messages.calls == [
        {
            "model": "claude-sonnet-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "say ok"}],
        }
    ]


def test_complete_with_image_returns_first_text_block(tmp_path: Path) -> None:
    image_path = tmp_path / "test.png"
    image_path.write_bytes(b"x")
    fake = _FakeAnthropic(_Response([_Block("text", "0.3")]))
    client = ClaudeClient(client=fake)
    assert client.complete_with_image("judge this", str(image_path)) == "0.3"


def test_complete_with_image_sends_base64_image_block_and_text(tmp_path: Path) -> None:
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(b"fake-jpeg-bytes")
    fake = _FakeAnthropic(_Response([_Block("text", "ok")]))
    client = ClaudeClient(model="claude-sonnet-5", client=fake)
    client.complete_with_image("describe this", str(image_path), max_tokens=16)
    assert fake.messages.calls == [
        {
            "model": "claude-sonnet-5",
            "max_tokens": 16,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "ZmFrZS1qcGVnLWJ5dGVz",
                            },
                        },
                        {"type": "text", "text": "describe this"},
                    ],
                }
            ],
        }
    ]


def test_complete_with_image_no_text_block_raises(tmp_path: Path) -> None:
    image_path = tmp_path / "test.png"
    image_path.write_bytes(b"x")
    fake = _FakeAnthropic(_Response([_Block("thinking", "...")], stop_reason="refusal"))
    client = ClaudeClient(client=fake)
    with pytest.raises(ValueError, match="no text block"):
        client.complete_with_image("judge this", str(image_path))
