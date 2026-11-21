"""OpenAI client seam: send a prompt to a model, get text back. No judge logic here."""

from __future__ import annotations

import os

import openai
from dotenv import load_dotenv
from openai.types.responses import EasyInputMessageParam, ResponseInputImageParam, ResponseInputTextParam

from aixeval.judge.image_encoding import guess_mime_type, read_image_base64


DEFAULT_MODEL = "gpt-5.5"
API_KEY_ENV_VAR = "OPENAI_API_KEY"


class OpenAIClient:
    """Thin wrapper over one OpenAI Responses API call. Prompt in, text out."""

    def __init__(self, *, model: str = DEFAULT_MODEL, client: openai.OpenAI | None = None) -> None:
        self._model = model
        self._client = client if client is not None else openai.OpenAI()

    @property
    def model(self) -> str:
        """The model id this client sends on every call. For attaching to a verdict."""
        return self._model

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        """Send one prompt, return the plain text reply.

        Uses the Responses API (``client.responses.create``), the surface
        OpenAI's own SDK README lists first for single-turn text requests.

        Parameters
        ----------
        prompt
            The full user input. No system/instructions framing here.
        max_tokens
            Passed as ``max_output_tokens``. Unlike the Claude client,
            this parameter name is not confirmed against a pinned
            reference doc for this SDK; verify against the installed
            ``openai`` package if a request rejects it.

        Returns
        -------
        str
            ``response.output_text``.

        Raises
        ------
        ValueError
            The reply had no text (e.g. blocked or empty).
        """
        response = self._client.responses.create(
            model=self._model,
            input=prompt,
            max_output_tokens=max_tokens,
        )
        text = response.output_text
        if not text:
            msg = "OpenAI reply had no text"
            raise ValueError(msg)
        return text

    def complete_with_image(self, prompt: str, image_path: str, *, max_tokens: int = 1024) -> str:
        """Send one prompt plus one image, return the plain text reply.

        Uses the Responses API's multimodal input shape: a user
        message with an ``input_text`` item and an ``input_image``
        item carrying a base64 data URL.

        Parameters
        ----------
        prompt
            The full user input.
        image_path
            Path to a local image file, base64-encoded into a data URL.
        max_tokens
            Passed as ``max_output_tokens``.

        Returns
        -------
        str
            ``response.output_text``.

        Raises
        ------
        ValueError
            The reply had no text (e.g. blocked or empty).
        """
        data_url = f"data:{guess_mime_type(image_path)};base64,{read_image_base64(image_path)}"
        text_item: ResponseInputTextParam = {"type": "input_text", "text": prompt}
        image_item: ResponseInputImageParam = {"type": "input_image", "image_url": data_url, "detail": "auto"}
        message: EasyInputMessageParam = {"role": "user", "content": [text_item, image_item]}
        response = self._client.responses.create(
            model=self._model,
            input=[message],
            max_output_tokens=max_tokens,
        )
        text = response.output_text
        if not text:
            msg = "OpenAI reply had no text"
            raise ValueError(msg)
        return text


def print_client_example() -> None:
    """Send one live prompt if the API key is set. Print how to set it otherwise."""
    load_dotenv()
    if not os.environ.get(API_KEY_ENV_VAR):
        print(f"Set {API_KEY_ENV_VAR} to run a live call. No key found; skipping.")
        return
    reply = OpenAIClient().complete("Reply with the single word: ok")
    print(f"model:  {DEFAULT_MODEL}")
    print("prompt: Reply with the single word: ok")
    print(f"reply:  {reply}")


if __name__ == "__main__":
    print_client_example()
