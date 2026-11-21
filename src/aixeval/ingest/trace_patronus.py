"""Map a Patronus/TRAIL-shaped export into the normalized trace model.

Verified against a real export: one top-level span per trace, with
the real run nested under ``child_spans``, not a flat span list like
Langfuse or raw OTLP. Span kind comes from
``span_attributes["openinference.span.kind"]``; the top-level ``span_kind``
field is always ``"Internal"`` and carries no information.

No agent identity: OpenInference has no dedicated agent-name attribute, and
this export's framework (``smolagents``) only puts a sub-agent's name in its
own prompt text or in the parent's ``smolagents.managed_agents.*.name``
attribute; neither is a general rule. ``agent`` is left ``None`` here
rather than guessed; a framework-specific extractor could be added later if
needed.
"""

from __future__ import annotations

from typing import Any

from aixeval.ingest.trace import Span, text_or_none, validate_spans


_KIND_BY_OI_KIND: dict[str, str] = {
    "AGENT": "agent",
    "LLM": "llm",
    "TOOL": "tool",
    "GUARDRAIL": "event",
}
_STATUS_BY_CODE: dict[str, str] = {"Ok": "ok", "Error": "error"}
_OI_KIND_ATTR = "openinference.span.kind"
_INPUT_ATTR = "input.value"
_OUTPUT_ATTR = "output.value"


def parse_patronus_trace(payload: dict[str, Any]) -> list[Span]:
    """Return the normalized spans for one Patronus/TRAIL export, validated."""
    roots = payload.get("spans")
    if not isinstance(roots, list) or not roots:
        msg = "Patronus payload has no spans list"
        raise ValueError(msg)
    spans: list[Span] = []
    for root in roots:
        _walk(root, spans)
    validate_spans(spans)
    return spans


def _walk(raw: object, spans: list[Span]) -> None:
    if not isinstance(raw, dict):
        msg = "Patronus span must be an object"
        raise ValueError(msg)
    span_id = raw.get("span_id")
    if not isinstance(span_id, str) or not span_id:
        msg = "Patronus span has no span_id"
        raise ValueError(msg)
    attrs = raw.get("span_attributes")
    attrs = attrs if isinstance(attrs, dict) else {}
    oi_kind = attrs.get(_OI_KIND_ATTR)
    kind = _KIND_BY_OI_KIND.get(oi_kind, "other") if isinstance(oi_kind, str) else "other"
    status_code = raw.get("status_code")

    spans.append(
        Span(
            id=span_id,
            kind=kind,
            parent_id=text_or_none(raw.get("parent_span_id")),
            agent=None,
            status=_STATUS_BY_CODE.get(status_code) if isinstance(status_code, str) else None,
            source_type=oi_kind if isinstance(oi_kind, str) else text_or_none(raw.get("span_name")),
            input=text_or_none(attrs.get(_INPUT_ATTR)),
            output=text_or_none(attrs.get(_OUTPUT_ATTR)),
            start=text_or_none(raw.get("timestamp")),
        )
    )
    for child in raw.get("child_spans") or []:
        _walk(child, spans)
