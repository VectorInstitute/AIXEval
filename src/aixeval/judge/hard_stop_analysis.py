"""Incident analysis: ask a judge model why a hard stop fired, citing trace steps.

An explanation only. It never feeds a metric, a grade or the card; the
hard stop itself is decided by the declared rule in ``scorecard.profile``.
Every step id the model cites is checked against the trace, so a reply
that points at a step that does not exist is rejected, not shown.
"""

from __future__ import annotations

import json
import re
from typing import Any

from aixeval.ingest.trace import Span
from aixeval.judge.provider import Provider


# Bump whenever _PROMPT_TEMPLATE changes; stored with every analysis.
PROMPT_VERSION = "hard-stop-analysis-v1"

_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

_PROMPT_TEMPLATE = """You are reviewing one finished AI agent run whose evaluation scorecard was \
capped by a HARD STOP. The hard stop is already decided by a fixed rule; do not re-grade it. \
Explain, from the trace, what the agent did that caused it.

HARD STOP FACTS (deterministic, from the scorer):
{facts}

METRIC RESULTS (grade 0 Unacceptable to 4 Excellent; N/A rows omitted):
{metrics}

EARLIER JUDGE RATIONALES (may be empty):
{rationales}

FINAL OUTPUT OF THE RUN:
{output}

TRACE STEPS (each starts with its step id in square brackets):
{steps}

Reply with one JSON object and nothing else, with exactly these keys:
  "root_cause": one or two sentences naming the specific action or decision that caused the hard stop.
  "responsible_steps": list of step ids (from the square brackets above) where that happened, most important first.
  "contributing_factors": list of short strings: earlier steps, inputs or reasoning that led there.
  "what_would_pass": one sentence: what the run should have done instead to avoid the hard stop.
  "confidence": number from 0 to 1: how clearly the trace shows this cause.
Cite only step ids that appear above. If the trace does not show the cause, say so in root_cause, \
use an empty responsible_steps list, and give a low confidence."""


def render_steps(spans: list[Span], *, max_field_chars: int = 600) -> str:
    """Render spans as id-tagged lines so a reply can cite them by id."""
    lines = []
    for span in spans:
        header = [f"[{span.id}]", span.kind]
        if span.agent:
            header.append(f"agent={span.agent}")
        if span.source_type and span.source_type.upper() != span.kind.upper():
            header.append(f"type={span.source_type}")
        if span.parent_id:
            header.append(f"parent={span.parent_id}")
        if span.status and span.status != "ok":
            header.append(f"status={span.status}")
        lines.append(" ".join(header))
        for label, text in (("input", span.input), ("output", span.output)):
            if text:
                clipped = text if len(text) <= max_field_chars else text[:max_field_chars] + " [...]"
                lines.append(f"  {label}: {clipped}")
    return "\n".join(lines)


def analyze_hard_stops(
    explanation: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    output: str | None,
    spans: list[Span] | None,
    rationale: dict[str, str],
    client: Provider,
    max_tokens: int = 4000,
) -> dict[str, Any]:
    """Ask ``client`` for a structured incident analysis of the fired hard stops.

    Parameters
    ----------
    explanation
        ``scorecard.hard_stops.explain_hard_stops`` output; must list at least one stop.
    rows
        The graded metric rows the card was built from.
    output
        The run's final output, if any.
    spans
        The run's trace; ``None`` or empty for a simple record.
    rationale
        Earlier judge rationales by metric id, passed through as context.
    client
        Any ``Provider``.
    max_tokens
        Headroom for reasoning models, as in the other judges.

    Returns
    -------
    dict
        ``root_cause``, ``responsible_steps``, ``contributing_factors``,
        ``what_would_pass``, ``confidence``, plus ``hard_stops`` (the stops
        analysed), ``prompt_version`` and ``model``.

    Raises
    ------
    ValueError
        No hard stop to analyse, or the reply was not valid JSON of the
        required shape, or it cited a step id that is not in the trace.
    """
    stops = [stop["hard_stop"] for stop in explanation.get("hard_stops", [])]
    if not stops:
        msg = "No hard stop fired; nothing to analyse."
        raise ValueError(msg)
    step_list = spans or []
    metrics = [
        {"metric_id": row["metric_id"], "grade": row["grade"], "raw_score": row.get("raw_score")}
        for row in rows
        if row.get("status") == "scored"
    ]
    prompt = _PROMPT_TEMPLATE.format(
        facts=json.dumps(explanation, indent=2, default=str),
        metrics=json.dumps(metrics, indent=2),
        rationales=json.dumps({key: value[:400] for key, value in rationale.items()}, indent=2) if rationale else "{}",
        output=output or "(none)",
        steps=render_steps(step_list) if step_list else "(no trace: this run is a simple record)",
    )
    reply = client.complete(prompt, max_tokens=max_tokens)
    analysis = parse_analysis(reply, valid_step_ids={span.id for span in step_list})
    return {**analysis, "hard_stops": stops, "prompt_version": PROMPT_VERSION, "model": client.model}


def parse_analysis(reply: str, *, valid_step_ids: set[str]) -> dict[str, Any]:
    """Parse and validate an analysis reply. Strict: no guessing."""
    data = _extract_json(reply)
    root_cause = data.get("root_cause")
    what_would_pass = data.get("what_would_pass")
    steps = data.get("responsible_steps")
    factors = data.get("contributing_factors")
    confidence = data.get("confidence")
    if not isinstance(root_cause, str) or not root_cause.strip():
        raise ValueError("Analysis reply has no root_cause text")
    if not isinstance(what_would_pass, str) or not what_would_pass.strip():
        raise ValueError("Analysis reply has no what_would_pass text")
    if not isinstance(steps, list) or not all(isinstance(step, str) for step in steps):
        raise ValueError("Analysis reply responsible_steps must be a list of step ids")
    if not isinstance(factors, list) or not all(isinstance(factor, str) for factor in factors):
        raise ValueError("Analysis reply contributing_factors must be a list of strings")
    if not isinstance(confidence, int | float) or isinstance(confidence, bool) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"Analysis reply confidence must be a number in [0, 1], got {confidence!r}")
    normalized = [_normalize_step_id(step) for step in steps]
    unknown = [step for step in normalized if step not in valid_step_ids]
    if unknown:
        raise ValueError(f"Analysis reply cites step id(s) not in the trace: {', '.join(unknown)}")
    return {
        "root_cause": root_cause.strip(),
        "responsible_steps": list(dict.fromkeys(normalized)),
        "contributing_factors": [factor.strip() for factor in factors if factor.strip()],
        "what_would_pass": what_would_pass.strip(),
        "confidence": float(confidence),
    }


def _normalize_step_id(step: str) -> str:
    """Bare id from a citation.

    The prompt shows steps as ``[a3]``; models often echo the brackets.
    """
    text = step.strip().strip("\"'")
    if len(text) >= 2 and text[0] == "[" and text[-1] == "]":
        text = text[1:-1].strip().strip("\"'")
    return text


def _extract_json(reply: str) -> dict[str, Any]:
    fenced = _FENCED_JSON.findall(reply)
    candidates = [fenced[-1]] if fenced else []
    start, end = reply.find("{"), reply.rfind("}")
    if start != -1 and end > start:
        candidates.append(reply[start : end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    msg = f"No JSON object in analysis reply: {reply[:300]!r}"
    raise ValueError(msg)
