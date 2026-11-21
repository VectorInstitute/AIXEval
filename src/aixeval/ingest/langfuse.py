"""Map a Langfuse-shaped JSON file onto a sealed record. File only. No SDK."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aixeval.ingest.load import load_json_object
from aixeval.ingest.record import validate_record


def ingest_langfuse(path: Path) -> dict[str, Any]:
    """Copy a few declared Langfuse paths. Do not guess gold or flags.

    Declared copies:

    - ``expected``: top-level or ``metadata.expected`` (our name only)
    - ``output``: top-level string ``output``; if missing, the last
      observation with ``type`` GENERATION and a string ``output``
    - ``stated_confidence``: ``metadata.stated_confidence``
    - ``latency_sec``: top-level ``latency`` (Langfuse seconds)
    - ``steps``: number of ``observations`` if that field is a list

    Vendor scores, ERROR levels, and ``gold`` are ignored. Missing stays
    missing. No live API.
    """
    return validate_record(langfuse_record_fields(load_json_object(path)))


def collect_langfuse_input_texts(payload: dict[str, Any]) -> list[str]:
    """Strings an audit may search: root ``input`` and each observation ``input``.

    A dict or list is JSON-encoded so a gold phrase inside structured
    input can still match. ``output`` fields are not collected.
    """
    texts: list[str] = []
    _append_input_text(texts, payload.get("input"))
    observations = payload.get("observations")
    if isinstance(observations, list):
        for item in observations:
            if isinstance(item, dict):
                _append_input_text(texts, item.get("input"))
    return texts


def _append_input_text(texts: list[str], value: object) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            texts.append(stripped)
        return
    if isinstance(value, dict | list):
        texts.append(json.dumps(value, default=str))


def langfuse_record_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Public: also reused by ``ingest.pipeline.ingest_any`` for Langfuse payloads."""
    record: dict[str, Any] = {}
    metadata = payload.get("metadata")
    meta = metadata if isinstance(metadata, dict) else {}

    expected = _optional_str(payload.get("expected")) or _optional_str(meta.get("expected"))
    if expected is not None:
        record["expected"] = expected

    output = _optional_str(payload.get("output"))
    observations = payload.get("observations")
    obs_list = observations if isinstance(observations, list) else []
    if output is None:
        output = _last_generation_output(obs_list)
    if output is not None:
        record["output"] = output

    confidence = meta.get("stated_confidence")
    if confidence is not None:
        record["stated_confidence"] = confidence

    latency = payload.get("latency")
    if latency is not None:
        record["latency_sec"] = latency

    if obs_list:
        record["steps"] = len(obs_list)
    return record


def _last_generation_output(observations: list[object]) -> str | None:
    """Last GENERATION span with string output. File order. Not SPAN/CHAIN."""
    found: str | None = None
    for item in observations:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if not isinstance(kind, str) or kind.strip().upper() != "GENERATION":
            continue
        text = _optional_str(item.get("output"))
        if text is not None:
            found = text
    return found


def _optional_str(value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def print_ingest_langfuse_examples() -> None:
    """Print how a Langfuse-shaped object becomes a sealed record."""
    print("ingest_langfuse copies declared paths only. No API. No guessed gold.")
    print()
    gist = {
        "output": "yes",
        "latency": 2.5,
        "metadata": {"expected": "yes"},
        "observations": [{"id": "o1"}, {"id": "o2"}],
    }
    record = validate_record(langfuse_record_fields(gist))
    print(f"  file:   {gist}")
    print(f"  record: {record}")


if __name__ == "__main__":
    print_ingest_langfuse_examples()
