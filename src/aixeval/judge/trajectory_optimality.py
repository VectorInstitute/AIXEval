"""Trajectory optimality verdict: ask a judge model about visible path waste.

Known sensitivity, first found live against a deep multi-agent trace (a
coordinator delegating to several sub-agents, each of which calls its own
tools and writes a prose summary of what they returned, before a final
agent summarizes those summaries): the judge can read that normal,
expected reporting pattern as redundancy, since the same facts genuinely
do appear more than once in the flattened trajectory text: once in a
tool's raw output, once in the sub-agent's own summary of it, once again
folded into the final synthesis. No step was actually repeated and no
work was redone; the judge still scored it 0.5 ("roughly 50% waste") on a
real trace shaped exactly like that.

Correction to an earlier version of this note: it previously claimed
shallow, single-agent traces hadn't shown this. They have: a plain
5-step single-agent trace (tool, tool, tool, tool, then the agent's own
final_output span restating "email sent") also drew a 0.5 verdict, the
judge reading the agent's final summary as a literal duplicate of the
tool call it was summarizing. So the real trigger isn't hierarchy depth;
it's any trace where a later step's text substantially restates an
earlier step's result, agent-to-tool or agent-to-agent alike, which
describes most well-run traces that end in a synthesizing step. Not
fixed (the prompt isn't adjusted for it), just documented, since a low
verdict on an otherwise well-run trace is often this known cause, not
necessarily a real finding of waste. Also a reminder that any single
verdict isn't perfectly reliable call to call (real, observed
non-determinism during live testing on this judge and others): two
structurally similar traces, scored in separate live calls, can land on
different sides of this (one fixture scored 1.0, a near-identical one
scored 0.5): a stored score is one draw, not a guarantee of reproducing
identically.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from aixeval.judge.config import build_client_from_env
from aixeval.judge.provider import Provider
from aixeval.judge.score_parse import parse_score


# Bump whenever _PROMPT_TEMPLATE changes; attached to every verdict
# so a stored trajectory_optimality_judge is traceable to the prompt
# that produced it.
PROMPT_VERSION = "trajectory-optimality-v1"


_KEY_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")

_PROMPT_TEMPLATE = """You are grading whether the STEPS below were a reasonably direct \
path to the OUTPUT, or whether they included clearly wasteful repetition.

STEPS:
{trajectory}

OUTPUT:
{output}

You cannot know the single objectively optimal path; do not invent one. \
Judge only what is visible: flag steps that are obviously redundant (the \
same action repeated with no new information gained), that contradict \
and redo earlier work, or that clearly did not contribute to OUTPUT. \
Do not penalize steps whose purpose is not obvious to you; the burden is \
on clear, visible waste, not on matching some imagined shortest path.

Think through your reasoning briefly, then end your reply with a single \
final line in exactly this form:
SCORE: <number>
where <number> is on a scale from 0 (repeated or clearly wasted steps) to \
1 (no visible redundancy). Use a decimal between 0 and 1 if only some \
steps were wasteful.
"""


def judge_trajectory_optimality(trajectory: str, output: str, client: Provider, *, max_tokens: int = 4000) -> float:
    """Ask ``client`` to grade whether ``trajectory`` wastefully reached ``output``.

    Matches the ``TrajectoryOptimalityJudge`` shape
    ``score_trajectory_optimality`` already accepts
    (``Callable[[str, str], float]``) once ``client`` is bound, e.g.
    ``functools.partial(judge_trajectory_optimality, client=my_client)``.

    Parameters
    ----------
    trajectory
        An ordered trace of the steps actually taken, e.g. from
        ``extract_trajectory_langfuse``.
    output
        What the system under test actually said: what the steps
        were working toward. A path can't be judged efficient without
        knowing the destination.
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
    prompt = _PROMPT_TEMPLATE.format(trajectory=trajectory, output=output)
    reply = client.complete(prompt, max_tokens=max_tokens)
    return parse_score(reply)


def print_trajectory_optimality_examples() -> None:
    """Judge one trajectory/output pair live, using whatever provider .env selects."""
    load_dotenv()
    if not any(os.environ.get(var) for var in _KEY_ENV_VARS):
        print("Set a provider API key (see .env.example) to run a live call. Skipping.")
        return
    client = build_client_from_env()
    trajectory = (
        "1. [TOOL] search -- input: capital of France -- output: Paris is the capital of France\n"
        "2. [TOOL] search -- input: capital of France -- output: Paris is the capital of France\n"
        "3. [GENERATION] llm -- output: Paris"
    )
    output = "Paris"
    score = judge_trajectory_optimality(trajectory, output, client)
    print(f"provider:   {type(client).__name__} ({client.model})")
    print(f"trajectory:\n{trajectory}")
    print(f"output:     {output}")
    print(f"score:      {score}")


if __name__ == "__main__":
    print_trajectory_optimality_examples()
