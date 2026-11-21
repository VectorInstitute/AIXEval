"""VLLMClient.complete: Chat Completions against a self-hosted vLLM server."""

from pathlib import Path

import pytest

from aixeval.judge.vllm_client import VLLMClient


class _Message:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str | None, finish_reason: str | None = None) -> None:
        self.message = _Message(content)
        self.finish_reason = finish_reason


class _Response:
    def __init__(self, choices: list[_Choice]) -> None:
        self.choices = choices


class _FakeCompletions:
    def __init__(self, response: _Response) -> None:
        self._response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _Response:
        self.calls.append(kwargs)
        return self._response


class _FakeChat:
    def __init__(self, response: _Response) -> None:
        self.completions = _FakeCompletions(response)


class _FakeOpenAI:
    def __init__(self, response: _Response) -> None:
        self.chat = _FakeChat(response)


def test_returns_message_content() -> None:
    fake = _FakeOpenAI(_Response([_Choice("0.8")]))
    client = VLLMClient(model="Qwen2.5-7B-Instruct", client=fake)
    assert client.complete("judge this") == "0.8"


def test_no_choices_raises() -> None:
    fake = _FakeOpenAI(_Response([]))
    client = VLLMClient(model="Qwen2.5-7B-Instruct", client=fake)
    with pytest.raises(ValueError, match="no choices"):
        client.complete("judge this")


def test_empty_content_raises() -> None:
    fake = _FakeOpenAI(_Response([_Choice("")]))
    client = VLLMClient(model="Qwen2.5-7B-Instruct", client=fake)
    with pytest.raises(ValueError, match="no text"):
        client.complete("judge this")


def test_empty_content_names_the_finish_reason() -> None:
    # A reasoning model that burns its token budget on hidden thinking
    # before emitting the SCORE: line stops with finish_reason="length";
    # naming it in the error turns a bare "no text" into an actionable one.
    fake = _FakeOpenAI(_Response([_Choice("", finish_reason="length")]))
    client = VLLMClient(model="Qwen2.5-7B-Instruct", client=fake)
    with pytest.raises(ValueError, match="finish_reason='length'"):
        client.complete("judge this")


def test_empty_content_with_no_finish_reason_still_raises_plainly() -> None:
    fake = _FakeOpenAI(_Response([_Choice("")]))
    client = VLLMClient(model="Qwen2.5-7B-Instruct", client=fake)
    with pytest.raises(ValueError, match=r"^vLLM reply had no text$"):
        client.complete("judge this")


def test_sends_chat_completions_shape() -> None:
    fake = _FakeOpenAI(_Response([_Choice("ok")]))
    client = VLLMClient(model="Qwen2.5-7B-Instruct", client=fake)
    client.complete("say ok", max_tokens=16)
    assert fake.chat.completions.calls == [
        {
            "model": "Qwen2.5-7B-Instruct",
            "messages": [{"role": "user", "content": "say ok"}],
            "max_tokens": 16,
            "temperature": 0,
        }
    ]


def test_send_temperature_false_omits_it_from_the_call() -> None:
    fake = _FakeOpenAI(_Response([_Choice("ok")]))
    client = VLLMClient(model="some-reasoning-model", client=fake, send_temperature=False)
    client.complete("say ok", max_tokens=16)
    assert fake.chat.completions.calls == [
        {
            "model": "some-reasoning-model",
            "messages": [{"role": "user", "content": "say ok"}],
            "max_tokens": 16,
        }
    ]


def test_model_property() -> None:
    client = VLLMClient(model="Qwen2.5-7B-Instruct", client=_FakeOpenAI(_Response([_Choice("ok")])))
    assert client.model == "Qwen2.5-7B-Instruct"


def test_complete_with_image_returns_message_content(tmp_path: Path) -> None:
    image_path = tmp_path / "test.png"
    image_path.write_bytes(b"x")
    fake = _FakeOpenAI(_Response([_Choice("0.3")]))
    client = VLLMClient(model="Qwen2.5-VL-7B-Instruct", client=fake)
    assert client.complete_with_image("judge this", str(image_path)) == "0.3"


def test_complete_with_image_no_choices_raises(tmp_path: Path) -> None:
    image_path = tmp_path / "test.png"
    image_path.write_bytes(b"x")
    fake = _FakeOpenAI(_Response([]))
    client = VLLMClient(model="Qwen2.5-VL-7B-Instruct", client=fake)
    with pytest.raises(ValueError, match="no choices"):
        client.complete_with_image("judge this", str(image_path))


def test_complete_with_image_sends_multimodal_chat_completions_shape(tmp_path: Path) -> None:
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(b"fake-jpeg-bytes")
    fake = _FakeOpenAI(_Response([_Choice("ok")]))
    client = VLLMClient(model="Qwen2.5-VL-7B-Instruct", client=fake)
    client.complete_with_image("describe this", str(image_path), max_tokens=16)
    assert fake.chat.completions.calls == [
        {
            "model": "Qwen2.5-VL-7B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,ZmFrZS1qcGVnLWJ5dGVz"},
                        },
                    ],
                }
            ],
            "max_tokens": 16,
            "temperature": 0,
        }
    ]


def test_no_api_key_needed_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    # vLLM's own server needs no key unless started with --api-key.
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    client = VLLMClient(model="Qwen2.5-7B-Instruct")
    assert client.model == "Qwen2.5-7B-Instruct"


def test_default_base_url_is_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    client = VLLMClient(model="Qwen2.5-7B-Instruct")
    # The openai SDK normalizes base_url with a trailing slash.
    assert str(client._client.base_url) == "http://localhost:8000/v1/"  # noqa: SLF001


def test_base_url_env_var_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_BASE_URL", "http://gpu-node-7:9000/v1")
    client = VLLMClient(model="Qwen2.5-7B-Instruct")
    assert str(client._client.base_url) == "http://gpu-node-7:9000/v1/"  # noqa: SLF001


def test_custom_base_url_env_var_reads_that_var_not_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.setenv("MY_VLLM_HOST", "http://other-host:8001/v1")
    client = VLLMClient(model="Qwen2.5-7B-Instruct", base_url_env_var="MY_VLLM_HOST")
    assert str(client._client.base_url) == "http://other-host:8001/v1/"  # noqa: SLF001
