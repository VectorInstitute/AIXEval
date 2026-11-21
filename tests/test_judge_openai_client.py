"""OpenAIClient.complete: one prompt in, output_text out. No live API here."""

from pathlib import Path

import pytest

from aixeval.judge.openai_client import OpenAIClient


class _Response:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


class _FakeResponses:
    def __init__(self, response: _Response) -> None:
        self._response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _Response:
        self.calls.append(kwargs)
        return self._response


class _FakeOpenAI:
    def __init__(self, response: _Response) -> None:
        self.responses = _FakeResponses(response)


def test_returns_output_text() -> None:
    fake = _FakeOpenAI(_Response("0.8"))
    client = OpenAIClient(client=fake)
    assert client.complete("judge this") == "0.8"


def test_empty_output_text_raises() -> None:
    fake = _FakeOpenAI(_Response(""))
    client = OpenAIClient(client=fake)
    with pytest.raises(ValueError, match="no text"):
        client.complete("judge this")


def test_sends_model_input_and_max_output_tokens() -> None:
    fake = _FakeOpenAI(_Response("ok"))
    client = OpenAIClient(model="gpt-5.5", client=fake)
    client.complete("say ok", max_tokens=16)
    assert fake.responses.calls == [{"model": "gpt-5.5", "input": "say ok", "max_output_tokens": 16}]


def test_model_property() -> None:
    client = OpenAIClient(model="gpt-5.5", client=_FakeOpenAI(_Response("ok")))
    assert client.model == "gpt-5.5"


def test_complete_with_image_returns_output_text(tmp_path: Path) -> None:
    image_path = tmp_path / "test.png"
    image_path.write_bytes(b"x")
    fake = _FakeOpenAI(_Response("0.3"))
    client = OpenAIClient(client=fake)
    assert client.complete_with_image("judge this", str(image_path)) == "0.3"


def test_complete_with_image_empty_output_text_raises(tmp_path: Path) -> None:
    image_path = tmp_path / "test.png"
    image_path.write_bytes(b"x")
    fake = _FakeOpenAI(_Response(""))
    client = OpenAIClient(client=fake)
    with pytest.raises(ValueError, match="no text"):
        client.complete_with_image("judge this", str(image_path))


def test_complete_with_image_sends_data_url_and_max_output_tokens(tmp_path: Path) -> None:
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(b"fake-jpeg-bytes")
    fake = _FakeOpenAI(_Response("ok"))
    client = OpenAIClient(model="gpt-5.5", client=fake)
    client.complete_with_image("describe this", str(image_path), max_tokens=16)
    assert fake.responses.calls == [
        {
            "model": "gpt-5.5",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "describe this"},
                        {
                            "type": "input_image",
                            "image_url": "data:image/jpeg;base64,ZmFrZS1qcGVnLWJ5dGVz",
                            "detail": "auto",
                        },
                    ],
                }
            ],
            "max_output_tokens": 16,
        }
    ]
