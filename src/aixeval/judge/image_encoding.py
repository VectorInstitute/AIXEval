"""Shared image-encoding helpers for provider clients' complete_with_image."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Literal


# The exact set Anthropic's Base64ImageSourceParam.media_type accepts --
# kept this narrow (not a plain str) so complete_with_image's SDK calls
# type-check without a cast.
MimeType = Literal["image/png", "image/jpeg", "image/webp", "image/gif"]

_MIME_TYPES: dict[str, MimeType] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def guess_mime_type(image_path: str) -> MimeType:
    """Infer the image MIME type from its file extension. Defaults to ``image/jpeg``."""
    return _MIME_TYPES.get(Path(image_path).suffix.lower(), "image/jpeg")


def read_image_base64(image_path: str) -> str:
    """Read the image file and return its base64-encoded bytes as an ASCII string."""
    return base64.standard_b64encode(Path(image_path).read_bytes()).decode("ascii")
