"""Map an OpenTelemetry OTLP JSON export into the normalized trace model.

Targets raw OTLP traces that never passed through Langfuse. Uses the
OTLP JSON field names (camelCase) and the GenAI semantic-convention
attributes ``gen_ai.operation.name`` and ``gen_ai.agent.name``. The GenAI
agent attributes are still provisional in the OTel spec, so this parser
is tied to today's attribute names and should be re-checked when that
spec changes.

Delegation text is not captured: input and output are None, so any metric
that needs handoff text reports N/A rather than guessing.
"""

from __future__ import annotations

from typing import Any

from aixeval.ingest.trace import Span, text_or_none, validate_spans


AGENT_NAME_ATTR = "gen_ai.agent.name"
OPERATION_ATTR = "gen_ai.operation.name"
_KIND_BY_OPERATION: dict[str, str] = {
    "invoke_agent": "agent",
    "execute_tool": "tool",
    "chat": "llm",
    "generate_content": "llm",
}
_STATUS_BY_CODE: dict[str, str | None] = {
    "STATUS_CODE_UNSET": None,
    "STATUS_CODE_OK": "ok",
    "STATUS_CODE_ERROR": "error",
    "0": None,
    "1": "ok",
    "2": "error",
}


def parse_otel_trace(payload: dict[str, Any]) -> list[Span]:
    """Return the normalized spans for one OTLP JSON trace, validated.

    Rejects payloads that contain more than one trace id, since one run is
    one trace.
    """
    resource_spans = payload.get("resourceSpans")
    if not isinstance(resource_spans, list):
        msg = "OTLP payload has no resourceSpans list"
        raise ValueError(msg)
    raw_spans = [
        span
        for resource in resource_spans
        for scope in _list(resource.get("scopeSpans"))
        for span in _list(scope.get("spans"))
    ]
    trace_ids = {span.get("traceId") for span in raw_spans if isinstance(span, dict)}
    if len(trace_ids) > 1:
        msg = f"OTLP payload holds {len(trace_ids)} trace ids; one run is one trace"
        raise ValueError(msg)

    spans: list[Span] = []
    for raw in raw_spans:
        spans.extend(_spans_from(raw))
    validate_spans(spans)
    return spans


def _spans_from(raw: object) -> list[Span]:
    if not isinstance(raw, dict):
        msg = "OTLP span must be an object"
        raise ValueError(msg)
    span_id = raw.get("spanId")
    if not isinstance(span_id, str) or not span_id:
        msg = "OTLP span has no spanId"
        raise ValueError(msg)
    attrs = _attributes(raw.get("attributes"))
    operation = attrs.get(OPERATION_ATTR)
    kind = _KIND_BY_OPERATION.get(operation, "other") if isinstance(operation, str) else "other"
    agent = attrs.get(AGENT_NAME_ATTR) if kind == "agent" and isinstance(attrs.get(AGENT_NAME_ATTR), str) else None
    parent = raw.get("parentSpanId")
    parent_id = parent if isinstance(parent, str) and parent else None

    spans = [
        Span(
            id=span_id,
            kind=kind,
            parent_id=parent_id,
            agent=agent,
            status=_status(raw.get("status")),
            source_type=operation if isinstance(operation, str) else raw.get("name"),
            start=text_or_none(raw.get("startTimeUnixNano")),
            end=text_or_none(raw.get("endTimeUnixNano")),
        )
    ]
    for index, event in enumerate(_list(raw.get("events"))):
        if isinstance(event, dict) and isinstance(event.get("name"), str):
            spans.append(
                Span(
                    id=f"{span_id}:event:{index}",
                    kind="event",
                    parent_id=span_id,
                    source_type=event["name"],
                    start=text_or_none(event.get("timeUnixNano")),
                )
            )
    return spans


def _attributes(raw: object) -> dict[str, Any]:
    """Flatten OTLP's ``[{key, value: {stringValue}}]`` list into a dict."""
    result: dict[str, Any] = {}
    for item in _list(raw):
        if not isinstance(item, dict) or not isinstance(item.get("key"), str):
            continue
        value = item.get("value")
        if isinstance(value, dict) and isinstance(value.get("stringValue"), str):
            result[item["key"]] = value["stringValue"]
    return result


def _status(raw: object) -> str | None:
    if not isinstance(raw, dict):
        return None
    code = raw.get("code")
    if isinstance(code, int | str):
        return _STATUS_BY_CODE.get(str(code))
    return None


def _list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []
