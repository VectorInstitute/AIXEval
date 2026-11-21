"""Evidence extraction from a raw dump, not the sealed record.

The generic mechanism is ``join_evidence_fields``: given a flat,
human-named dict and a list of field names, join whichever are present.
Per-format navigation into that flat shape is the caller's job; only
genuinely general, widely-shared formats get a shipped navigator here
(Langfuse, a real third-party platform, matching ``ingest_langfuse``).
A one-off internal dump shape does not: write a small flattening
function for it at the call site and pass the result straight to
``join_evidence_fields``, the same way ``extract_evidence_langfuse``
does below.

``extract_trajectory_langfuse`` walks the same ``observations`` list
but pulls a different shape of information: an ordered trace of
steps taken, for judging path quality (see ``trajectory_optimality``),
not evidence for judging answer faithfulness (see ``grounding``).

tau-bench's own equivalent, ``extract_tool_calls_taubench``, lives in
``aixeval.ingest.trace_taubench`` instead of here: it shares
private helpers with that module's ``parse_taubench_trace``, so keeping
both in one place avoided a second implementation of the same
arguments-shape handling (GPT-4o's JSON-string vs. Sonnet's native dict).

``extract_tool_calls_langfuse`` walks the same list again, filtered
to just ``TOOL``-type observations and reshaped as structured
``{"tool": ..., "args": ...}`` dicts, for deterministic, no-LLM
comparison against a declared gold tool-call list (see
``tool_call_accuracy``), not free text for a judge to read.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from aixeval.ingest.langfuse import collect_langfuse_input_texts


def join_evidence_fields(source: dict[str, Any], field_names: Sequence[str]) -> str | None:
    """Join whichever named fields are present in ``source``.

    Parameters
    ----------
    source
        A flat dict of candidate evidence fields, human-named, not raw
        JSON keys from the original dump.
    field_names
        Which of ``source``'s keys to include, and in what order. A
        future field-picker (UI or LLM-assisted) only ever needs to
        produce this list.

    Returns
    -------
    str | None
        The joined text, or ``None`` if none of ``field_names`` were
        present with usable content.
    """
    texts: list[str] = []
    for name in field_names:
        _append_field_text(texts, source.get(name))
    return "\n\n".join(texts) if texts else None


def _append_field_text(texts: list[str], value: object) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            texts.append(stripped)
        return
    if isinstance(value, dict | list):
        texts.append(json.dumps(value, default=str))


def extract_evidence_langfuse(payload: dict[str, Any]) -> str | None:
    """Pull the input texts a Langfuse-shaped trace's agent was given.

    Reuses ``collect_langfuse_input_texts`` (already used for the
    gold-in-input audit): the same definition of "what the agent saw"
    both times, not a second implementation of the same walk.

    Returns
    -------
    str | None
        The joined input texts, or ``None`` if there were none.
    """
    texts = collect_langfuse_input_texts(payload)
    return "\n\n".join(texts) if texts else None


def extract_trajectory_langfuse(payload: dict[str, Any]) -> str | None:
    """Pull an ordered, readable trace of the steps a Langfuse-shaped run took.

    Walks the same ``observations`` list ``extract_evidence_langfuse``
    reads, but pulls ``type``/``name``/``input``/``output`` per step
    instead of just ``input``: a trace of what happened, not evidence
    for what the answer should say.

    Returns
    -------
    str | None
        One numbered line per observation, or ``None`` if there were
        none.
    """
    observations = payload.get("observations")
    if not isinstance(observations, list):
        return None
    lines: list[str] = []
    for index, item in enumerate(observations, start=1):
        if not isinstance(item, dict):
            continue
        line = _format_step(index, item)
        if line is not None:
            lines.append(line)
    return "\n".join(lines) if lines else None


def _format_step(index: int, observation: dict[str, Any]) -> str | None:
    kind = observation.get("type")
    name = observation.get("name")
    step_input = observation.get("input")
    step_output = observation.get("output")
    if kind is None and name is None and step_input is None and step_output is None:
        return None
    label = f"[{kind}]" if kind else "[step]"
    who = f" {name}" if name else ""
    parts = [f"{index}. {label}{who}"]
    if step_input is not None:
        parts.append(f"input: {_as_text(step_input)}")
    if step_output is not None:
        parts.append(f"output: {_as_text(step_output)}")
    return " -- ".join(parts)


def _as_text(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def extract_tool_calls_langfuse(payload: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Pull the actual tool calls a Langfuse-shaped run made, as structured data.

    Filters the same ``observations`` list ``extract_trajectory_langfuse``
    reads down to ``type == "TOOL"`` entries only, reshaped as
    ``{"tool": <name>, "args": <input>}``, ready to compare against a
    declared ``gold_tool_calls`` list with no LLM involved.

    Returns
    -------
    list[dict] | None
        One entry per tool call, in order, or ``None`` if there were
        none. A tool call with no usable ``name`` is skipped; a missing
        ``input`` becomes ``{}``, never ``None``, so downstream
        comparison never has to special-case it.
    """
    observations = payload.get("observations")
    if not isinstance(observations, list):
        return None
    calls: list[dict[str, Any]] = []
    for item in observations:
        if not isinstance(item, dict) or item.get("type") != "TOOL":
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            continue
        args = item.get("input")
        calls.append({"tool": name, "args": args if isinstance(args, dict) else {}})
    return calls or None


def print_evidence_examples() -> None:
    """Print how the generic join and the Langfuse navigator behave."""
    print("join_evidence_fields: generic, given a flat dict, join selected fields.")
    flat = {"chart_text": "axis ticks: 1QFY22, 2QFY22", "caption": "Fig. 10: Mexico Sales"}
    print(f"  evidence: {join_evidence_fields(flat, ['chart_text', 'caption'])}")
    print()
    print("extract_evidence_langfuse: reuses collect_langfuse_input_texts.")
    langfuse = {"input": "search: Mexico sales trend", "observations": [{"input": "tool result: 195 -> 225"}]}
    print(f"  evidence: {extract_evidence_langfuse(langfuse)}")
    print()
    print("extract_trajectory_langfuse: same observations list, different fields.")
    trajectory_payload = {
        "observations": [
            {"type": "SPAN", "name": "agent", "input": "Capital of France?", "output": "planning"},
            {"type": "GENERATION", "name": "llm", "output": "Paris"},
        ]
    }
    print(f"  trajectory:\n{extract_trajectory_langfuse(trajectory_payload)}")
    print()
    print("extract_tool_calls_langfuse: same observations, filtered + reshaped, no LLM.")
    tool_call_payload = {
        "observations": [
            {"type": "SPAN", "name": "agent", "input": "weather in Paris in F", "output": "planning"},
            {"type": "TOOL", "name": "get_weather", "input": {"city": "Paris"}, "output": {"temp_c": 20}},
            {"type": "GENERATION", "name": "llm", "output": "It's about 68F in Paris."},
        ]
    }
    print(f"  tool calls: {extract_tool_calls_langfuse(tool_call_payload)}")


if __name__ == "__main__":
    print_evidence_examples()
