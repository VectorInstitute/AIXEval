"""Map a Langfuse export's observations into the normalized trace model.

Field names are the ones verified against a real export: ``type``,
``id``, ``parentObservationId``, ``level``, ``statusMessage``,
``input``, ``output``, ``startTime``, ``endTime``, and the agent
identity under ``metadata["attributes.gen_ai.agent.name"]``.
"""

from __future__ import annotations

from typing import Any

from aixeval.ingest.trace import Span, text_or_none, validate_spans


AGENT_NAME_KEY = "attributes.gen_ai.agent.name"
_KIND_BY_TYPE: dict[str, str] = {
    "AGENT": "agent",
    "TOOL": "tool",
    "GENERATION": "llm",
    "EVENT": "event",
    "GUARDRAIL": "event",
}
_STATUS_BY_LEVEL: dict[str, str] = {"ERROR": "error", "DEFAULT": "ok"}
_NAME_PREFIX = "agent_run ["
_NAME_SUFFIX = "]"


def parse_langfuse_trace(payload: dict[str, Any]) -> list[Span]:
    """Return the normalized spans for one Langfuse export, validated."""
    observations = payload.get("observations")
    if not isinstance(observations, list):
        msg = "Langfuse payload has no observations list"
        raise ValueError(msg)
    spans = [_span(item) for item in observations]
    validate_spans(spans)
    return spans


def _span(item: object) -> Span:
    if not isinstance(item, dict):
        msg = "Langfuse observation must be an object"
        raise ValueError(msg)
    span_id = item.get("id")
    if not isinstance(span_id, str) or not span_id:
        msg = "Langfuse observation has no id"
        raise ValueError(msg)
    observation_type = item.get("type")
    level = item.get("level")
    kind = _KIND_BY_TYPE.get(observation_type, "other") if isinstance(observation_type, str) else "other"
    # For an event-kind span, Langfuse's own type values ("EVENT"/"GUARDRAIL")
    # only ever say "this is an event": the specific kind (fault_injected,
    # guardrail check outcome, etc.) lives in the observation's name, same
    # convention OTel's span-event "name" field already uses. Every other
    # kind keeps source_type=type, unchanged.
    source_type = _event_name(item) if kind == "event" else observation_type
    return Span(
        id=span_id,
        kind=kind,
        parent_id=text_or_none(item.get("parentObservationId")),
        agent=_agent_identity(item) if kind == "agent" else None,
        status=_STATUS_BY_LEVEL.get(level) if isinstance(level, str) else None,
        source_type=source_type if isinstance(source_type, str) else None,
        input=text_or_none(item.get("input")),
        output=text_or_none(item.get("output")),
        start=text_or_none(item.get("startTime")),
        end=text_or_none(item.get("endTime")),
    )


def _agent_identity(item: dict[str, Any]) -> str | None:
    """Return the metadata agent attribute, falling back to the name template."""
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get(AGENT_NAME_KEY)
        if isinstance(value, str) and value:
            return value
    name = item.get("name")
    if isinstance(name, str) and name.startswith(_NAME_PREFIX) and name.endswith(_NAME_SUFFIX):
        inner = name[len(_NAME_PREFIX) : -len(_NAME_SUFFIX)]
        return inner or None
    return None


def _event_name(item: dict[str, Any]) -> str | None:
    """Return an event span's own name, e.g. ``"fault_injected"``/``"guardrail"``."""
    name = item.get("name")
    return name if isinstance(name, str) and name else None
