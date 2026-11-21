"""Metric relevance suggester: ask a judge which metrics are worth running.

Deliberately separate from the deterministic "will this metric actually
score given the fields on hand" gate: that question is already answered
by each scorer's own N/A logic inside ``score_metrics()``. This answers a
different question: is this metric even *meaningful* for this kind of run?
For example ``cross_modal_hallucination`` is not meaningful for a pure-text
customer-support transcript even if every field it needs happened to be
declared; ``delegation_accuracy`` is not meaningful for a single-agent run.

A UI combines both signals: suggest-relevant (this module) intersected
with scorable-now (``score_metrics()``'s own status) decides what gets
pre-checked, but either signal alone is shown too, so a user can still
opt into a relevant-but-not-yet-scorable metric by supplying more fields,
or a scorable-but-not-suggested one if they disagree with the judge.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence

from dotenv import load_dotenv

from aixeval.judge.config import build_client_from_env
from aixeval.judge.provider import Provider
from aixeval.scorecard.definitions import definition_for_any_metric
from aixeval.scorecard.metrics import ALL_METRIC_IDS


# Bump whenever _PROMPT_TEMPLATE changes.
PROMPT_VERSION = "metric-suggester-v1"

_KEY_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")

_RELEVANT_LINE = re.compile(r"^RELEVANT:\s*([a-z0-9_]+)\s*[-:—]\s*(.+)$", re.IGNORECASE)

_PROMPT_TEMPLATE = """You are helping choose which evaluation metrics are worth running on one \
saved agent run.

RUN DESCRIPTION (as typed by the person uploading it):
{description}
{facts_block}
Below is the full catalog of metrics this toolkit can compute, one per \
line as `metric_id: what it measures`:
{catalog}

Pick only the metrics that are meaningful for this specific run. Ignore \
whether the data needed to compute it happens to be available right now \
-- that is checked separately, elsewhere. Judge relevance only: for \
example a pure-text customer-support transcript has no meaningful \
cross_modal_hallucination or clip_score; a single-agent run has no \
meaningful delegation_accuracy.{facts_trust_note}

Reply with one line per metric you judge relevant, in exactly this form, \
nothing else before or after:
RELEVANT: <metric_id> - <one short reason>
"""


def suggest_relevant_metrics(
    description: str,
    client: Provider,
    *,
    metric_ids: Sequence[str] = ALL_METRIC_IDS,
    max_tokens: int = 6000,
    source_kind: str | None = None,
    has_trajectory: bool | None = None,
    num_distinct_agents: int | None = None,
    has_image: bool | None = None,
) -> dict[str, str]:
    """Ask ``client`` which of ``metric_ids`` are meaningful for this run.

    Parameters
    ----------
    description
        A short free-text summary of the run (domain, what the agent(s)
        did, and so on): what the person uploading it typed, not
        verified against the actual data.
    client
        Any ``Provider``.
    metric_ids
        Which metric ids to offer. Defaults to the full catalog
        (``ALL_METRIC_IDS``: auto-wired and opt-in both).
    max_tokens
        Passed to ``client.complete``. The catalog is long (44 lines by
        default), so this needs more headroom than a single-score judge.
        Default of 6000 is set from live testing against the Vector
        reasoning model (``Qwen3_6-35B-A3B``): at 2048 it spends the
        whole budget on internal reasoning before emitting any
        ``RELEVANT:`` lines and ``client.complete`` raises ``ValueError:
        ... had no text``. A faster/non-reasoning provider may need far
        less, but 6000 is a safe default across providers.
    source_kind, has_trajectory, num_distinct_agents, has_image
        Objective facts checked directly against the ingested record/
        spans; not re-derived from ``description``, which the caller
        may have misstated or left incomplete (e.g. forgot to mention a
        run is multi-agent, or that an image is attached). ``None`` means
        not known/not checked, and that fact is simply left out of the
        prompt rather than asserted as false. Pass these whenever an
        ``IngestResult`` already exists to check (the UI always does
        once something is ingested); omit them only when suggesting
        against a bare description with nothing ingested yet.

    Returns
    -------
    dict[str, str]
        ``{metric_id: reason}`` for every metric the model flagged
        relevant. A metric it did not mention is simply not suggested,
        not an error, and not a claim that the metric cannot be scored.

    Raises
    ------
    ValueError
        The reply had no parseable ``RELEVANT:`` line naming a known
        metric id.
    """
    catalog = "\n".join(f"{metric_id}: {definition_for_any_metric(metric_id)}" for metric_id in metric_ids)
    facts_block = _format_facts(
        source_kind=source_kind,
        has_trajectory=has_trajectory,
        num_distinct_agents=num_distinct_agents,
        has_image=has_image,
    )
    # Only tell the model to trust RUN FACTS over the description when a
    # RUN FACTS section actually exists; otherwise this would be a
    # dangling reference to a section that was never rendered.
    facts_trust_note = (
        " Where RUN FACTS and RUN DESCRIPTION disagree, trust RUN FACTS: it is "
        "checked directly against the ingested data; the description is only what "
        "the person typed and may be incomplete or wrong."
        if facts_block
        else ""
    )
    prompt = _PROMPT_TEMPLATE.format(
        description=description, facts_block=facts_block, facts_trust_note=facts_trust_note, catalog=catalog
    )
    reply = client.complete(prompt, max_tokens=max_tokens)
    return _parse_relevant(reply, known_ids=set(metric_ids))


def _format_facts(
    *,
    source_kind: str | None,
    has_trajectory: bool | None,
    num_distinct_agents: int | None,
    has_image: bool | None,
) -> str:
    """Render only the facts supplied; an unset fact is omitted, never guessed."""
    lines: list[str] = []
    if source_kind is not None:
        lines.append(f"- source kind: {source_kind}")
    if has_trajectory is not None:
        lines.append(f"- has a step-by-step trajectory: {'yes' if has_trajectory else 'no'}")
    if num_distinct_agents is not None:
        shape = "multi-agent" if num_distinct_agents > 1 else "single-agent"
        lines.append(f"- distinct agents observed: {num_distinct_agents} ({shape})")
    if has_image is not None:
        lines.append(f"- reference image attached: {'yes' if has_image else 'no'}")
    if not lines:
        return ""
    body = "\n".join(lines)
    return f"\nRUN FACTS (checked directly against the ingested data):\n{body}\n"


def _parse_relevant(reply: str, *, known_ids: set[str]) -> dict[str, str]:
    """Collect each well-formed ``RELEVANT: <id> - <reason>`` line naming a known id."""
    suggestions: dict[str, str] = {}
    for line in reply.strip().splitlines():
        match = _RELEVANT_LINE.match(line.strip())
        if match is None:
            continue
        metric_id = match.group(1).lower()
        reason = match.group(2).strip()
        if metric_id in known_ids and reason:
            suggestions[metric_id] = reason
    if not suggestions:
        msg = f"No parseable RELEVANT: line naming a known metric in judge reply: {reply!r}"
        raise ValueError(msg)
    return suggestions


def print_metric_suggester_examples() -> None:
    """Suggest metrics for one example run, live, using whatever .env selects."""
    load_dotenv()
    if not any(os.environ.get(var) for var in _KEY_ENV_VARS):
        print("Set a provider API key (see .env.example) to run a live call. Skipping.")
        return
    client = build_client_from_env()
    description = (
        "A single customer-support agent answered one text question about "
        "a billing dispute. No images, no sub-agents, no tool-call delegation."
    )
    suggestions = suggest_relevant_metrics(description, client)
    print(f"provider: {type(client).__name__} ({client.model})")
    print(f"description: {description}")
    for metric_id, reason in sorted(suggestions.items()):
        print(f"  {metric_id}: {reason}")


if __name__ == "__main__":
    print_metric_suggester_examples()
