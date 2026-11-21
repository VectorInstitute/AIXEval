"""Map an Inspect AI ``.eval`` log into the normalized trace model.

A ``.eval`` file is a Zstd-compressed zip archive (``header.json`` plus one
``samples/<id>_epoch_<n>.json`` per sample/epoch pair), not a plain JSON
payload like Langfuse/OTel/Patronus: ``read_inspect_eval`` handles the
archive layer, ``parse_inspect_sample`` maps one already-parsed sample dict
to spans. ``zipfile``'s stdlib decompressor doesn't support Zstd on this
project's Python version; ``zipfile_zstd`` is imported for its side effect
of registering it (compression method 93) with ``zipfile`` directly, so
``zipfile.ZipFile(...).read(...)`` works normally afterward.

Built from ``sample["messages"]`` (Inspect's flattened, fully-resolved chat
history), not the richer raw ``sample["events"]`` stream; events store
conversation content via index-range references into a shared attachment
buffer this parser doesn't resolve, while messages already carry plain
text or a structured tool call directly. A real, stated asymmetry: per-step
timing, exact sandbox exit codes, and ``span_begin``/``span_end`` nesting
are not captured here, only each turn's text and each tool call's name,
arguments, and returned text, same "stated honestly, not hidden"
discipline as every other parser's documented gap in this module.

Single-agent only: nothing in a Inspect sample names a sub-agent the way
Langfuse's ``gen_ai.agent.name`` or an OTel ``invoke_agent`` span does, so
every span here is ``kind="llm"`` or ``kind="tool"``, never ``"agent"``:
``derive_delegations``/``derive_faults`` correctly find nothing, and the
four MAS metrics correctly report N/A rather than a guessed delegation.
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any

from aixeval.ingest.trace import Span, validate_spans


# Decompression caps. A .eval file is an untrusted upload in the UI; a tiny
# zip can expand to gigabytes (a "zip bomb"). Real Inspect samples are well
# under these limits; raise them via the keyword arguments if one is not.
MAX_SAMPLE_BYTES = 256 * 1024 * 1024
MAX_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_SAMPLE_ENTRIES = 100_000


def read_inspect_eval(
    data: bytes,
    *,
    max_sample_bytes: int = MAX_SAMPLE_BYTES,
    max_total_bytes: int = MAX_TOTAL_BYTES,
    max_entries: int = MAX_SAMPLE_ENTRIES,
) -> list[dict[str, Any]]:
    """Return every sample/epoch record a ``.eval`` file contains.

    Sorted by ``(sample id, epoch)`` so callers get a stable, predictable
    order: one dataset sample run for several epochs appears as
    consecutive entries, not shuffled by zip member order.

    Decompressed sizes are enforced while reading, not taken from the zip
    header (which an attacker controls), so a crafted archive cannot
    exhaust memory.

    Parameters
    ----------
    data
        Raw bytes of the ``.eval`` file.
    max_sample_bytes
        Largest decompressed size allowed for one ``samples/*.json`` entry.
    max_total_bytes
        Largest decompressed size allowed across all sample entries.
    max_entries
        Most ``samples/*.json`` entries allowed.

    Raises
    ------
    ValueError
        Not a zip archive, it has no ``samples/*.json`` entries, or it
        exceeds one of the size/entry caps.
    """
    import zipfile_zstd  # noqa: F401,PLC0415 (side effect: registers Zstd with zipfile)

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = [n for n in archive.namelist() if n.startswith("samples/") and n.endswith(".json")]
            if not names:
                msg = "no samples/*.json entries found in this .eval file"
                raise ValueError(msg)
            if len(names) > max_entries:
                msg = f".eval file has {len(names)} sample entries, more than the limit of {max_entries}"
                raise ValueError(msg)
            samples = []
            total = 0
            for name in names:
                raw = _read_capped(archive, name, max_sample_bytes)
                total += len(raw)
                if total > max_total_bytes:
                    msg = f".eval samples decompress to more than the limit of {max_total_bytes} bytes"
                    raise ValueError(msg)
                samples.append(json.loads(raw))
    except zipfile.BadZipFile as exc:
        msg = "not a valid .eval file (expected a zip archive)"
        raise ValueError(msg) from exc

    samples.sort(key=lambda s: (s.get("id"), s.get("epoch", 0)))
    return samples


def _read_capped(archive: zipfile.ZipFile, name: str, limit: int) -> bytes:
    """Read one member, stopping one byte past ``limit``.

    The zip header's declared size is not trusted.
    """
    with archive.open(name) as member:
        raw = member.read(limit + 1)
    if len(raw) > limit:
        msg = f".eval entry {name!r} decompresses to more than the limit of {limit} bytes"
        raise ValueError(msg)
    return raw


def parse_inspect_sample(sample: dict[str, Any]) -> list[Span]:
    """Return normalized spans for one Inspect sample's message trace.

    One span per assistant turn (``kind="llm"``, ``output`` is its text)
    and per tool response (``kind="tool"``, ``input`` is the triggering
    call's ``function(arguments)``, ``output`` is the returned text).
    System/user messages (the task prompt) are not spans, same as every
    other parser here never keeping the raw prompt as a step.

    Raises
    ------
    ValueError
        ``sample`` has no ``messages`` list.
    """
    messages = sample.get("messages")
    if not isinstance(messages, list):
        msg = "Inspect sample has no messages list"
        raise ValueError(msg)

    tool_calls_by_id = _index_tool_calls(messages)
    spans: list[Span] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            spans.append(Span(id=f"msg{index}", kind="llm", output=_message_text(message.get("content"))))
        elif role == "tool":
            call_id = message.get("tool_call_id")
            call = tool_calls_by_id.get(call_id) if isinstance(call_id, str) else None
            spans.append(
                Span(
                    id=f"msg{index}",
                    kind="tool",
                    input=_tool_call_text(call) if call else None,
                    output=_message_text(message.get("content")),
                )
            )
    validate_spans(spans)
    return spans


def _index_tool_calls(messages: list[Any]) -> dict[str, dict[str, Any]]:
    """Map each assistant tool call's id to the call itself, across all turns."""
    index: dict[str, dict[str, Any]] = {}
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict) and isinstance(call.get("id"), str):
                index[call["id"]] = call
    return index


def _tool_call_text(call: dict[str, Any]) -> str | None:
    function = call.get("function")
    if not isinstance(function, str):
        return None
    arguments = call.get("arguments")
    return f"{function}({arguments})" if arguments else function


def _message_text(content: object) -> str | None:
    """Extract plain text from ``content``, a plain string or a content-block list.

    List form (seen on turns with internal reasoning) mixes ``"reasoning"``
    blocks (hidden thinking) with ``"text"`` blocks (what was actually
    said); only the latter is kept, matching the record's existing
    reasoning/output distinction elsewhere in this toolkit.
    """
    if isinstance(content, str):
        return content or None
    if isinstance(content, list):
        parts = [
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        ]
        joined = "".join(parts).strip()
        return joined or None
    return None
