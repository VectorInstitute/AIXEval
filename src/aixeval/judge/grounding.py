"""Grounding verdict: ask a judge model whether output stays faithful to evidence."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from aixeval.judge.config import build_client_from_env
from aixeval.judge.provider import Provider
from aixeval.judge.score_parse import parse_score


# Bump whenever _PROMPT_TEMPLATE changes; attached to every verdict
# so a stored grounding_judge is traceable to the prompt that produced it.
PROMPT_VERSION = "grounding-v1"


_KEY_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")

_PROMPT_TEMPLATE = """You are grading whether an AI system's OUTPUT stays faithful to the given EVIDENCE.

Faithful means every claim in OUTPUT is directly supported by EVIDENCE. \
Unsupported claims, invented facts, or numbers/details not present in \
EVIDENCE count against faithfulness, even if they sound plausible.

EVIDENCE:
{evidence}

OUTPUT:
{output}

Think through your reasoning briefly, then end your reply with a single \
final line in exactly this form:
SCORE: <number>
where <number> is on a scale from 0 (completely unsupported or \
contradicted by the evidence) to 1 (fully supported). Use a decimal \
between 0 and 1 if only partially supported.
"""


def judge_grounding(evidence: str, output: str, client: Provider, *, max_tokens: int = 4000) -> float:
    """Ask ``client`` to grade faithfulness of ``output`` to ``evidence`` on ``[0, 1]``.

    Matches the ``GroundingJudge`` shape ``score_grounding`` already
    accepts (``Callable[[str, str], float]``) once ``client`` is bound,
    e.g. ``functools.partial(judge_grounding, client=my_client)``.

    Parameters
    ----------
    evidence
        The evidence text the output should be faithful to.
    output
        What the system under test actually said.
    client
        Any ``Provider``: Claude, OpenAI, Gemini, or Vector.
    max_tokens
        Passed to ``client.complete``. Needs real headroom: a
        reasoning-capable model can spend most of this budget on its
        own reasoning before ever emitting the final ``SCORE:`` line.
        Default of 4000 matches the value confirmed live against the
        Vector reasoning model (``Qwen3_6-35B-A3B``); 1024 was observed
        to fail with ``ValueError: ... had no text`` on real traces.

    Returns
    -------
    float
        The parsed score, in ``[0, 1]``.

    Raises
    ------
    ValueError
        The reply had no parseable, in-range ``SCORE:`` line.
    """
    prompt = _PROMPT_TEMPLATE.format(evidence=evidence, output=output)
    reply = client.complete(prompt, max_tokens=max_tokens)
    return parse_score(reply)


def print_grounding_examples() -> None:
    """Judge one evidence/output pair live, using whatever provider .env selects."""
    load_dotenv()
    if not any(os.environ.get(var) for var in _KEY_ENV_VARS):
        print("Set a provider API key (see .env.example) to run a live call. Skipping.")
        return
    client = build_client_from_env()
    evidence = "Mexico sales rose from 195 in 3QFY22 to 225 in 2QFY23."
    output = "Mexico sales grew from 3QFY22 to 2QFY23."
    score = judge_grounding(evidence, output, client)
    print(f"provider: {type(client).__name__} ({client.model})")
    print(f"evidence: {evidence}")
    print(f"output:   {output}")
    print(f"score:    {score}")


if __name__ == "__main__":
    print_grounding_examples()
