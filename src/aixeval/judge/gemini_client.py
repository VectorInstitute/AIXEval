"""Gemini client seam: send a prompt to a model, get text back. No judge logic here."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from aixeval.judge.image_encoding import guess_mime_type


DEFAULT_MODEL = "gemini-flash-latest"
API_KEY_ENV_VAR = "GEMINI_API_KEY"


class GeminiClient:
    """Thin wrapper over one Gemini generate_content call. Prompt in, text out."""

    def __init__(self, *, model: str = DEFAULT_MODEL, client: genai.Client | None = None) -> None:
        self._model = model
        self._client = client if client is not None else genai.Client()

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
            Passed as ``config.max_output_tokens``.

        Returns
        -------
        str
            ``response.text``.

        Raises
        ------
        ValueError
            The reply had no text (e.g. blocked by safety settings).
        """
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=max_tokens, temperature=0),
        )
        text = response.text
        if not text:
            msg = "Gemini reply had no text"
            raise ValueError(msg)
        return text

    def complete_with_image(self, prompt: str, image_path: str, *, max_tokens: int = 1024) -> str:
        """Send one prompt plus one image, return the plain text reply.

        Parameters
        ----------
        prompt
            The full user input.
        image_path
            Path to a local image file, read as raw bytes and sent as
            an inline ``types.Part`` alongside ``prompt``.
        max_tokens
            Passed as ``config.max_output_tokens``.

        Returns
        -------
        str
            ``response.text``.

        Raises
        ------
        ValueError
            The reply had no text (e.g. blocked by safety settings).
        """
        image_part = types.Part.from_bytes(data=Path(image_path).read_bytes(), mime_type=guess_mime_type(image_path))
        # list[Any]: the SDK's generate_content overloads for a mixed
        # [Part, str] list don't unify cleanly under mypy's invariant
        # list generics (see its own `Content | ContentDict | str |
        # Image | ... | list[...]` union); Any is the least-bad
        # escape hatch here, not a blanket suppression.
        contents: list[Any] = [image_part, prompt]
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(max_output_tokens=max_tokens, temperature=0),
        )
        text = response.text
        if not text:
            msg = "Gemini reply had no text"
            raise ValueError(msg)
        return text


def print_client_example() -> None:
    """Send one live prompt if the API key is set. Print how to set it otherwise."""
    load_dotenv()
    if not os.environ.get(API_KEY_ENV_VAR):
        print(f"Set {API_KEY_ENV_VAR} to run a live call. No key found; skipping.")
        return
    reply = GeminiClient().complete("Reply with the single word: ok")
    print(f"model:  {DEFAULT_MODEL}")
    print("prompt: Reply with the single word: ok")
    print(f"reply:  {reply}")


if __name__ == "__main__":
    print_client_example()
