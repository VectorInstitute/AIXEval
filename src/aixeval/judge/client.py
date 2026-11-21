"""LLM client seam: send a prompt to Claude, get text back. No grounding logic here."""

from __future__ import annotations

import os

import anthropic
from anthropic.types import ImageBlockParam, MessageParam, TextBlockParam
from dotenv import load_dotenv

from aixeval.judge.image_encoding import guess_mime_type, read_image_base64


DEFAULT_MODEL = "claude-sonnet-5"
API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"


class ClaudeClient:
    """Thin wrapper over one Claude call. Prompt in, text out. No judge logic."""

    def __init__(self, *, model: str = DEFAULT_MODEL, client: anthropic.Anthropic | None = None) -> None:
        self._model = model
        self._client = client if client is not None else anthropic.Anthropic()

    @property
    def model(self) -> str:
        """The model id this client sends on every call. For attaching to a verdict."""
        return self._model

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        """Send one prompt, return the first text block.

        Parameters
        ----------
        prompt
            The full user message. No system prompt or judge framing here.
        max_tokens
            Passed straight to the API.

        Returns
        -------
        str
            The first text content block in the reply.

        Raises
        ------
        ValueError
            The reply had no text block (e.g. only a refusal).
        """
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        for block in response.content:
            if block.type == "text":
                return block.text
        msg = f"Claude reply had no text block (stop_reason={response.stop_reason!r})"
        raise ValueError(msg)

    def complete_with_image(self, prompt: str, image_path: str, *, max_tokens: int = 1024) -> str:
        """Send one prompt plus one image, return the first text block.

        Parameters
        ----------
        prompt
            The full user message.
        image_path
            Path to a local image file (PNG/JPEG/WEBP/GIF), base64-encoded
            and sent as an image content block alongside ``prompt``.
        max_tokens
            Passed straight to the API.

        Returns
        -------
        str
            The first text content block in the reply.

        Raises
        ------
        ValueError
            The reply had no text block (e.g. only a refusal).
        """
        image_block: ImageBlockParam = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": guess_mime_type(image_path),
                "data": read_image_base64(image_path),
            },
        }
        text_block: TextBlockParam = {"type": "text", "text": prompt}
        message: MessageParam = {"role": "user", "content": [image_block, text_block]}
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[message],
        )
        for block in response.content:
            if block.type == "text":
                return block.text
        msg = f"Claude reply had no text block (stop_reason={response.stop_reason!r})"
        raise ValueError(msg)


def print_client_example() -> None:
    """Send one live prompt if the API key is set. Print how to set it otherwise."""
    load_dotenv()
    if not os.environ.get(API_KEY_ENV_VAR):
        print(f"Set {API_KEY_ENV_VAR} to run a live call. No key found; skipping.")
        return
    reply = ClaudeClient().complete("Reply with the single word: ok")
    print(f"model:  {DEFAULT_MODEL}")
    print("prompt: Reply with the single word: ok")
    print(f"reply:  {reply}")


if __name__ == "__main__":
    print_client_example()
