"""Pairwise preference verdict: ask a judge model which of two responses is better.

Named ``pairwise_preference``, not ``win_rate``/``Elo``, deliberately:
a *cross-system aggregate* ("LLM-judge / win-rate / Elo") rolls many
pairwise verdicts into a win percentage or an Elo rating, then ranks
that against other systems' aggregates; that's not a single record's
score. This module builds only the per-record building block one level
below that: one pairwise verdict for one record's ``output`` against
a declared ``baseline_output``. Rolling many of these into a win-rate
or Elo rating, and ranking that against a pool, is future work this
module does not attempt.

Position-bias correction (Zheng et al., Judging LLM-as-a-Judge with
MT-Bench and Chatbot Arena, NeurIPS 2023): an LLM judge is measurably
more likely to prefer whichever response is presented first. That
paper's formula averages judged preference over both presentation
orders using raw token probabilities; this module does not have uniform logprob
access across all four providers, so it approximates the same
correction with two full judge calls (once with ``output`` shown
first, once with ``baseline_output`` shown first) and averages the
two preference scores. Twice the cost of a single-call judge, by
design, not an oversight.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from aixeval.judge.config import build_client_from_env
from aixeval.judge.provider import Provider
from aixeval.judge.score_parse import parse_score


# Bump whenever _PROMPT_TEMPLATE changes; attached to every verdict
# so a stored pairwise_preference_judge is traceable to the prompt
# that produced it.
PROMPT_VERSION = "pairwise-preference-judge-v1"


_KEY_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")

_PROMPT_TEMPLATE = """You are comparing two AI responses to decide which one better \
accomplishes the task.
{reference_block}
RESPONSE A:
{response_a}

RESPONSE B:
{response_b}

Judge overall quality and task success: correctness, completeness, and \
helpfulness, not length or style alone. {reference_hint}

Think through your reasoning briefly, then end your reply with a single \
final line in exactly this form:
SCORE: <number>
where <number> is 1 if RESPONSE A is clearly better, 0 if RESPONSE B is \
clearly better, and 0.5 if they are about equally good. Use a decimal \
between 0 and 1 for a partial preference.
"""


def judge_pairwise_preference(
    output: str,
    baseline_output: str,
    client: Provider,
    *,
    reference: str | None = None,
    max_tokens: int = 4000,
) -> float:
    """Ask ``client`` which of ``output``/``baseline_output`` is better, on ``[0, 1]``.

    Matches the ``PairwisePreferenceJudge`` shape
    ``score_pairwise_preference`` already accepts
    (``Callable[[str, str], float]``) once ``client`` (and optionally
    ``reference``) is bound, e.g.
    ``functools.partial(judge_pairwise_preference, client=my_client)``.

    Makes two judge calls, not one, to correct for position bias:
    once presenting ``output`` as RESPONSE A, once with the order
    swapped. The two preference scores are averaged. 1.0 means
    ``output`` won both presentations outright; 0.0 means it lost
    both; 0.5 is a genuine tie or a bias-cancelling split decision.

    Parameters
    ----------
    output
        The candidate response being scored.
    baseline_output
        The opponent response ``output`` is compared against.
    client
        Any ``Provider``: Claude, OpenAI, Gemini, or Vector.
    reference
        An optional gold/reference answer. When given, the judge is
        told to use it as a guide to correctness, not a strict
        template, the same spirit as AlpacaEval-style reference-
        anchored pairwise judging. When absent, the judge falls back
        to general helpfulness and correctness.
    max_tokens
        Passed to ``client.complete`` on each of the two calls. Needs
        real headroom: a reasoning-capable model can spend most of
        this budget on its own reasoning before ever emitting the
        final ``SCORE:`` line. Default of 4000 matches the value
        confirmed live against the Vector reasoning model
        (``Qwen3_6-35B-A3B``); 1024 was observed to fail with
        ``ValueError: ... had no text`` on a real call.

    Returns
    -------
    float
        The position-bias-corrected preference score, in ``[0, 1]``.

    Raises
    ------
    ValueError
        Either reply had no parseable, in-range ``SCORE:`` line.
    """
    forward = _compare(output, baseline_output, client, reference=reference, max_tokens=max_tokens)
    backward = _compare(baseline_output, output, client, reference=reference, max_tokens=max_tokens)
    # In the backward call, `output` is RESPONSE B, so its implied
    # preference there is (1 - backward)'s score for RESPONSE A.
    return (forward + (1.0 - backward)) / 2.0


def _compare(response_a: str, response_b: str, client: Provider, *, reference: str | None, max_tokens: int) -> float:
    if reference is not None:
        reference_block = f"\nREFERENCE ANSWER (for context on what a good answer contains):\n{reference}\n"
        reference_hint = "Use the reference answer as a guide to correctness, not a strict template."
    else:
        reference_block = ""
        reference_hint = "No reference answer is available; judge general correctness and helpfulness."
    prompt = _PROMPT_TEMPLATE.format(
        reference_block=reference_block, response_a=response_a, response_b=response_b, reference_hint=reference_hint
    )
    reply = client.complete(prompt, max_tokens=max_tokens)
    return parse_score(reply)


def print_pairwise_preference_examples() -> None:
    """Judge one output/baseline pair live, using whatever provider .env selects."""
    load_dotenv()
    if not any(os.environ.get(var) for var in _KEY_ENV_VARS):
        print("Set a provider API key (see .env.example) to run a live call. Skipping.")
        return
    client = build_client_from_env()
    output = "The capital of France is Paris, a city on the Seine known for the Eiffel Tower."
    baseline_output = "Paris."
    score = judge_pairwise_preference(output, baseline_output, client)
    print(f"provider:        {type(client).__name__} ({client.model})")
    print(f"output (A):      {output}")
    print(f"baseline (B):    {baseline_output}")
    print(f"score:           {score}")


if __name__ == "__main__":
    print_pairwise_preference_examples()
