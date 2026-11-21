"""Turn one saved JSON object into a sealed record. Slice 1: simple JSON only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aixeval.ingest.load import load_json_object, load_json_suite
from aixeval.ingest.record import RECORD_FIELDS, validate_record


def ingest_dump(path: Path) -> dict[str, Any]:
    """Load one JSON object and copy known fields onto a sealed record.

    ``choices`` (letter → text) is translator input, not a record field.
    If ``output`` is a letter in that map, it becomes the choice text.
    Everything else is omitted. Unknown record keys fail validation.
    """
    return validate_record(simple_record_fields(load_json_object(path)))


def ingest_suite(path: Path) -> list[dict[str, Any]]:
    """Load many simple JSON objects (array or ``.jsonl``) into sealed records.

    Each row uses the same copy rules as ``ingest_dump``. One bad row fails
    the whole file.
    """
    records: list[dict[str, Any]] = []
    for index, payload in enumerate(load_json_suite(path)):
        try:
            records.append(validate_record(simple_record_fields(payload)))
        except ValueError as exc:
            msg = f"suite item {index}: {exc}"
            raise ValueError(msg) from exc
    return records


def simple_record_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Copy sealed-record keys. Expand a letter ``output`` when ``choices`` exist.

    Public (not a leading-underscore helper): also reused directly by
    ``aixeval.ingest.pipeline.ingest_any`` for its simple-dump
    fallback path, so the one copy-known-fields rule lives in one place.
    """
    record: dict[str, Any] = {}
    for key in RECORD_FIELDS:
        if key in payload and payload[key] is not None:
            record[key] = payload[key]

    choices = payload.get("choices")
    output = record.get("output")
    if isinstance(choices, dict) and isinstance(output, str):
        expanded = _expand_letter(output, choices)
        if expanded is not None:
            record["output"] = expanded
    return record


def _expand_letter(output: str, choices: dict[object, object]) -> str | None:
    """Return choice text if ``output`` is a single-letter key in ``choices``."""
    token = output.strip()
    if len(token) != 1:
        return None
    for key, text in choices.items():
        if str(key).strip().upper() == token.upper() and isinstance(text, str):
            return text
    return None


def print_ingest_dump_examples() -> None:
    """Print how a simple JSON object becomes a sealed record."""
    print("ingest_dump copies known names onto a sealed record.")
    print("  choices is not a record field; it only expands a letter output.")
    print("  Missing keys stay missing. Unknown record keys fail validation.")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("Already our names.", {"expected": "yes", "output": "yes"}),
        (
            "Letter output plus choices.",
            {"expected": "Paris", "output": "A", "choices": {"A": "Paris", "B": "Lyon"}},
        ),
        ("No gold.", {"output": "yes"}),
    )
    for story, payload in examples:
        record = validate_record(simple_record_fields(payload))
        print(story)
        print(f"  file:   {payload}")
        print(f"  record: {record}")
        print()


def print_ingest_suite_examples() -> None:
    """Print how a two-row suite becomes two sealed records."""
    print("ingest_suite: JSON array or .jsonl. Same copy rules as ingest_dump.")
    rows: list[dict[str, Any]] = [
        {"expected": "yes", "output": "yes"},
        {"expected": "Paris", "output": "A", "choices": {"A": "Paris"}},
    ]
    print(f"  file rows: {rows}")
    print(f"  records:   {[validate_record(simple_record_fields(row)) for row in rows]}")


if __name__ == "__main__":
    print_ingest_dump_examples()
    print_ingest_suite_examples()
