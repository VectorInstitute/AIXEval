"""GeminiClient.complete: one prompt in, response.text out. No live API here."""

from pathlib import Path

import pytest
from google.genai import types

from aixeval.judge.gemini_client import GeminiClient


class _Response:
    def __init__(self, text: str | None) -> None:
        self.text = text


class _FakeModels:
    def __init__(self, response: _Response) -> None:
        self._response = response
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs: object) -> _Response:
        self.calls.append(kwargs)
        return self._response


class _FakeGenAI:
    def __init__(self, response: _Response) -> None:
        self.models = _FakeModels(response)


def test_returns_response_text() -> None:
    fake = _FakeGenAI(_Response("0.8"))
    client = GeminiClient(client=fake)
    assert client.complete("judge this") == "0.8"


def test_empty_text_raises() -> None:
    fake = _FakeGenAI(_Response(""))
    client = GeminiClient(client=fake)
    with pytest.raises(ValueError, match="no text"):
        client.complete("judge this")


def test_blocked_reply_with_none_text_raises() -> None:
    fake = _FakeGenAI(_Response(None))
    client = GeminiClient(client=fake)
    with pytest.raises(ValueError, match="no text"):
        client.complete("judge this")


def test_sends_model_contents_and_max_output_tokens() -> None:
    fake = _FakeGenAI(_Response("ok"))
    client = GeminiClient(model="gemini-flash-latest", client=fake)
    client.complete("say ok", max_tokens=16)
    assert fake.models.calls == [
        {
            "model": "gemini-flash-latest",
            "contents": "say ok",
            "config": types.GenerateContentConfig(max_output_tokens=16, temperature=0),
        }
    ]


def test_model_property() -> None:
    client = GeminiClient(model="gemini-flash-latest", client=_FakeGenAI(_Response("ok")))
    assert client.model == "gemini-flash-latest"


def test_complete_with_image_returns_response_text(tmp_path: Path) -> None:
    image_path = tmp_path / "test.png"
    image_path.write_bytes(b"x")
    fake = _FakeGenAI(_Response("0.3"))
    client = GeminiClient(client=fake)
    assert client.complete_with_image("judge this", str(image_path)) == "0.3"


def test_complete_with_image_empty_text_raises(tmp_path: Path) -> None:
    image_path = tmp_path / "test.png"
    image_path.write_bytes(b"x")
    fake = _FakeGenAI(_Response(""))
    client = GeminiClient(client=fake)
    with pytest.raises(ValueError, match="no text"):
        client.complete_with_image("judge this", str(image_path))


def test_complete_with_image_sends_image_part_prompt_and_max_output_tokens(tmp_path: Path) -> None:
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(b"fake-jpeg-bytes")
    fake = _FakeGenAI(_Response("ok"))
    client = GeminiClient(model="gemini-flash-latest", client=fake)
    client.complete_with_image("describe this", str(image_path), max_tokens=16)
    assert fake.models.calls == [
        {
            "model": "gemini-flash-latest",
            "contents": [
                types.Part.from_bytes(data=b"fake-jpeg-bytes", mime_type="image/jpeg"),
                "describe this",
            ],
            "config": types.GenerateContentConfig(max_output_tokens=16, temperature=0),
        }
    ]
