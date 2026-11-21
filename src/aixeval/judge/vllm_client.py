"""A generic self-hosted vLLM server, reached over its OpenAI-compatible API.

Uses the ``openai`` SDK with a configurable ``base_url``. The server
URL comes from the environment so it can be localhost, a LAN host, or
a tunnel. vLLM needs no API key unless the operator started it with
``--api-key``; authentication here is optional.
"""

from __future__ import annotations

import os
from typing import Any

import openai
from dotenv import load_dotenv
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionUserMessageParam,
)

from aixeval.judge.image_encoding import guess_mime_type, read_image_base64


BASE_URL_ENV_VAR = "VLLM_BASE_URL"
API_KEY_ENV_VAR = "VLLM_API_KEY"
DEFAULT_BASE_URL = "http://localhost:8000/v1"
# vLLM's own OpenAI-compatible server accepts any value here when it was
# started without --api-key; the openai SDK itself requires a non-empty
# string regardless, so this fills that gap rather than being a real secret.
_UNUSED_API_KEY = "vllm-no-api-key-set"


class VLLMClient:
    """Thin wrapper over one Chat Completions call to a self-hosted vLLM server.

    vLLM's server implements the OpenAI Chat Completions surface, not
    the newer Responses API ``OpenAIClient`` uses.
    """

    def __init__(
        self,
        *,
        model: str,
        client: openai.OpenAI | None = None,
        base_url_env_var: str = BASE_URL_ENV_VAR,
        api_key_env_var: str = API_KEY_ENV_VAR,
        send_temperature: bool = True,
    ) -> None:
        """Build a client for a self-hosted vLLM server.

        Parameters
        ----------
        model
            Whatever model id the server was started with. vLLM serves
            exactly one model per process, so there is no toolkit-wide
            default the way hosted providers have one.
        client
            An already-built ``openai.OpenAI`` to use instead of
            constructing one from ``base_url_env_var``/``api_key_env_var``
            (e.g. for tests, or a non-standard auth setup).
        base_url_env_var
            Which environment variable names the server's base URL.
            Read at construction time, default ``http://localhost:8000/v1``
            if unset.
        api_key_env_var
            Which environment variable names the API key, if the server
            was started with ``--api-key``. Unset is fine: a placeholder
            value is sent instead, since the ``openai`` SDK requires a
            non-empty string but vLLM without ``--api-key`` ignores it.
        send_temperature
            Defaults on, ``temperature=0`` for reproducibility. Set
            ``False`` for a model/server combination that rejects an
            explicit temperature, same gotcha already documented for
            OpenAI's and Vector's reasoning-tier models.
        """
        self._model = model
        self._send_temperature = send_temperature
        base_url = os.environ.get(base_url_env_var, DEFAULT_BASE_URL)
        api_key = os.environ.get(api_key_env_var) or _UNUSED_API_KEY
        self._client = client if client is not None else openai.OpenAI(base_url=base_url, api_key=api_key)

    @property
    def model(self) -> str:
        """The model id this client sends on every call. For attaching to a verdict."""
        return self._model

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        """Send one prompt, return the plain text reply.

        Parameters
        ----------
        prompt
            The full user input.
        max_tokens
            Passed as ``max_tokens`` on the Chat Completions call.

        Returns
        -------
        str
            ``response.choices[0].message.content``.

        Raises
        ------
        ValueError
            The reply had no text (e.g. no choices, or empty content).
        """
        kwargs: dict[str, Any] = {"temperature": 0} if self._send_temperature else {}
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            **kwargs,
        )
        return _text_from_chat_response(response)

    def complete_with_image(self, prompt: str, image_path: str, *, max_tokens: int = 1024) -> str:
        """Send one prompt plus one image via Chat Completions' multimodal content.

        Whether this actually works depends on the deployed model being
        vision-capable (e.g. a Qwen-VL or LLaVA checkpoint); a text-only
        model errors on this call instead, surfacing to the caller like
        any other judge-call failure, not silently.

        Parameters
        ----------
        prompt
            The full user input.
        image_path
            Path to a local image file, base64-encoded into a data URL.
        max_tokens
            Passed as ``max_tokens`` on the Chat Completions call.

        Returns
        -------
        str
            ``response.choices[0].message.content``.

        Raises
        ------
        ValueError
            The reply had no text (e.g. no choices, or empty content).
        """
        data_url = f"data:{guess_mime_type(image_path)};base64,{read_image_base64(image_path)}"
        text_part: ChatCompletionContentPartTextParam = {"type": "text", "text": prompt}
        image_part: ChatCompletionContentPartImageParam = {"type": "image_url", "image_url": {"url": data_url}}
        message: ChatCompletionUserMessageParam = {"role": "user", "content": [text_part, image_part]}
        kwargs: dict[str, Any] = {"temperature": 0} if self._send_temperature else {}
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[message],
            max_tokens=max_tokens,
            **kwargs,
        )
        return _text_from_chat_response(response)


def _text_from_chat_response(response: Any) -> str:
    """Return ``choices[0].message.content``, or raise with the stop reason."""
    if not response.choices:
        msg = "vLLM reply had no choices"
        raise ValueError(msg)
    choice = response.choices[0]
    text = choice.message.content
    if not text:
        reason = getattr(choice, "finish_reason", None)
        msg = "vLLM reply had no text"
        if reason:
            msg = f"{msg} (finish_reason={reason!r})"
        raise ValueError(msg)
    return text


def print_client_example() -> None:
    """Send one live prompt if JUDGE_MODEL_VLLM is set and a server is reachable."""
    load_dotenv()
    model = os.environ.get("JUDGE_MODEL_VLLM")
    if not model:
        print("Set JUDGE_MODEL_VLLM (and optionally VLLM_BASE_URL/VLLM_API_KEY) to run a live call. Skipping.")
        return
    base_url = os.environ.get(BASE_URL_ENV_VAR, DEFAULT_BASE_URL)
    reply = VLLMClient(model=model).complete("Reply with the single word: ok")
    print(f"model:  {model}")
    print(f"base:   {base_url}")
    print("prompt: Reply with the single word: ok")
    print(f"reply:  {reply}")


if __name__ == "__main__":
    print_client_example()
