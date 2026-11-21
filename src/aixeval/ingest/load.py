"""Load one JSON object or a suite (array / JSONL)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_object(path: Path) -> dict[str, Any]:
    """Read UTF-8 JSON. The root must be an object."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = "ingest accepts one JSON object, not a list or other value"
        raise ValueError(msg)
    return payload


def load_json_suite(path: Path) -> list[dict[str, Any]]:
    """Read many objects: ``.jsonl`` (one object per line) or a JSON array."""
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() == ".jsonl":
        return _load_jsonl(text)
    payload = json.loads(text)
    if not isinstance(payload, list):
        msg = "ingest_suite accepts a JSON array or a .jsonl file"
        raise ValueError(msg)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            msg = f"suite item {index} must be a JSON object"
            raise ValueError(msg)
        rows.append(item)
    return rows


def _load_jsonl(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = f"jsonl line {line_no} is not valid JSON"
            raise ValueError(msg) from exc
        if not isinstance(item, dict):
            msg = f"jsonl line {line_no} must be a JSON object"
            raise ValueError(msg)
        rows.append(item)
    return rows
