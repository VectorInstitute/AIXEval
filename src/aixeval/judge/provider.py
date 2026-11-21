"""Provider protocol: one shape every judge LLM client must match."""

from __future__ import annotations

from typing import Protocol


class Provider(Protocol):
    """Anything that can turn one prompt into one text reply."""

    @property
    def model(self) -> str:
        """The model id this client sends on every call."""
        ...

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        """Send one prompt, return the first text reply."""
        ...

    def complete_with_image(self, prompt: str, image_path: str, *, max_tokens: int = 1024) -> str:
        """Send one prompt plus one image, return the first text reply."""
        ...
