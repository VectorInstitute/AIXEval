"""guess_mime_type/read_image_base64: small, pure helpers, no SDK involved."""

import base64
from pathlib import Path

from aixeval.judge.image_encoding import guess_mime_type, read_image_base64


def test_guesses_known_extensions() -> None:
    assert guess_mime_type("/data/photo.png") == "image/png"
    assert guess_mime_type("/data/photo.jpg") == "image/jpeg"
    assert guess_mime_type("/data/photo.jpeg") == "image/jpeg"
    assert guess_mime_type("/data/photo.webp") == "image/webp"
    assert guess_mime_type("/data/photo.gif") == "image/gif"


def test_unknown_extension_defaults_to_jpeg() -> None:
    assert guess_mime_type("/data/photo.bmp") == "image/jpeg"


def test_extension_matching_is_case_insensitive() -> None:
    assert guess_mime_type("/data/PHOTO.PNG") == "image/png"


def test_reads_and_base64_encodes_the_file(tmp_path: Path) -> None:
    image_path = tmp_path / "test.png"
    image_path.write_bytes(b"not-a-real-png-but-fine-for-this-test")
    encoded = read_image_base64(str(image_path))
    assert base64.standard_b64decode(encoded) == b"not-a-real-png-but-fine-for-this-test"
