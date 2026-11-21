"""Abstention verdict: ask a judge model whether output appropriately declines."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from aixeval.judge.config import build_client_from_env
from aixeval.judge.provider import Provider
from aixeval.judge.score_parse import parse_score


# Bump whenever _PROMPT_TEMPLATE changes; attached to every verdict
# so a stored abstention_judge is traceable to the prompt that produced it.
PROMPT_VERSION = "abstention-v1"


_KEY_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")

_PROMPT_TEMPLATE = """You are grading whether an AI system's OUTPUT appropriately declines \
to answer, rather than attempting a confident answer.

OUTPUT:
{output}

An appropriate abstention says, in substance, that it cannot determine \
the answer (for example stating uncertainty, declining, or asking for \
more information), rather than committing to a specific answer it is \
not positioned to know is correct.

Think through your reasoning briefly, then end your reply with a single \
final line in exactly this form:
SCORE: <number>
where <number> is 1 if OUTPUT appropriately declines/defers, and 0 if it \
attempts a confident answer instead. Use a decimal between 0 and 1 only \
if OUTPUT partially hedges while still committing to an answer.
"""


def judge_abstention(output: str, client: Provider, *, max_tokens: int = 4000) -> float:
    """Ask ``client`` to grade whether ``output`` appropriately abstains, on ``[0, 1]``.

    Matches the ``AbstentionJudge`` shape ``score_abstention`` already
    accepts (``Callable[[str], float]``) once ``client`` is bound, e.g.
    ``functools.partial(judge_abstention, client=my_client)``.

    Parameters
    ----------
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
        to fail with ``ValueError: ... had no text`` on a real call.

    Returns
    -------
    float
        The parsed score, in ``[0, 1]``.

    Raises
    ------
    ValueError
        The reply had no parseable, in-range ``SCORE:`` line.
    """
    prompt = _PROMPT_TEMPLATE.format(output=output)
    reply = client.complete(prompt, max_tokens=max_tokens)
    return parse_score(reply)


def print_abstention_examples() -> None:
    """Judge one output live, using whatever provider .env selects."""
    load_dotenv()
    if not any(os.environ.get(var) for var in _KEY_ENV_VARS):
        print("Set a provider API key (see .env.example) to run a live call. Skipping.")
        return
    client = build_client_from_env()
    output = "I don't have enough information to determine that with confidence."
    score = judge_abstention(output, client)
    print(f"provider: {type(client).__name__} ({client.model})")
    print(f"output:   {output}")
    print(f"score:    {score}")


if __name__ == "__main__":
    print_abstention_examples()
