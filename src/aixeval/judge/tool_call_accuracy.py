"""Tool-call accuracy verdict (judge path): ask a judge model about tool-use.

This is one of two independent ways to produce a ``tool_call_accuracy_judge``
verdict; see ``aixeval.scorecard.tool_call_accuracy`` for the other
(deterministic) path and why they're interchangeable.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from aixeval.judge.config import build_client_from_env
from aixeval.judge.provider import Provider
from aixeval.judge.score_parse import parse_score


# Bump whenever _PROMPT_TEMPLATE changes; attached to every verdict
# so a stored tool_call_accuracy_judge is traceable to the prompt that
# produced it.
PROMPT_VERSION = "tool-call-accuracy-judge-v1"


_KEY_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")

_PROMPT_TEMPLATE = """You are grading whether the STEPS below used the correct tools, with \
correct arguments, to reach OUTPUT, compared against GOLD PLAN, a \
description of what should have happened.

GOLD PLAN:
{gold_tool_plan}

STEPS:
{trajectory}

OUTPUT:
{output}

Grade tool use only, not whether OUTPUT is correct. A tool called with \
wrong or incomplete arguments, a required tool never called, or an \
unnecessary tool called instead of the right one all count against \
correctness, even if OUTPUT happens to be right anyway.

Think through your reasoning briefly, then end your reply with a single \
final line in exactly this form:
SCORE: <number>
where <number> is on a scale from 0 (wrong or missing tool use) to 1 \
(every required tool used correctly). Use a decimal between 0 and 1 if \
only some of the required tool use was correct.
"""


def judge_tool_call_accuracy(
    trajectory: str, gold_tool_plan: str, output: str, client: Provider, *, max_tokens: int = 4000
) -> float:
    """Ask ``client`` to grade tool-use correctness against ``gold_tool_plan``.

    Matches the ``ToolCallAccuracyJudge`` shape
    ``score_tool_call_accuracy`` already accepts
    (``Callable[[str, str, str], float]``) once ``client`` is bound,
    e.g. ``functools.partial(judge_tool_call_accuracy, client=my_client)``.

    Parameters
    ----------
    trajectory
        An ordered trace of the steps actually taken, e.g. from
        ``extract_trajectory_langfuse``.
    gold_tool_plan
        A declared, one-sentence-or-so description of which tools
        should have been used, same authoring shape as ``expected``,
        not a structured tool-call list (that's the deterministic
        path's job).
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
    prompt = _PROMPT_TEMPLATE.format(gold_tool_plan=gold_tool_plan, trajectory=trajectory, output=output)
    reply = client.complete(prompt, max_tokens=max_tokens)
    return parse_score(reply)


def print_tool_call_accuracy_examples() -> None:
    """Judge one trajectory/gold-plan/output triple live, using .env's provider."""
    load_dotenv()
    if not any(os.environ.get(var) for var in _KEY_ENV_VARS):
        print("Set a provider API key (see .env.example) to run a live call. Skipping.")
        return
    client = build_client_from_env()
    trajectory = (
        "1. [SPAN] agent -- input: weather in Paris in F -- output: planning\n"
        '2. [TOOL] get_weather -- input: {"city": "Paris"} -- output: {"temp_c": 20}\n'
        "3. [GENERATION] llm -- output: It's about 68F in Paris."
    )
    gold_tool_plan = "Should call get_weather for the city, then convert_temperature from Celsius to Fahrenheit."
    output = "It's about 68F in Paris."
    # max_tokens=4000: the default 1024 is tight for a reasoning-capable
    # model (e.g. Vector's Qwen3) to reach the final SCORE: line at all.
    score = judge_tool_call_accuracy(trajectory, gold_tool_plan, output, client, max_tokens=4000)
    print(f"provider:        {type(client).__name__} ({client.model})")
    print(f"gold_tool_plan:  {gold_tool_plan}")
    print(f"trajectory:\n{trajectory}")
    print(f"output:          {output}")
    print(f"score:           {score}")


if __name__ == "__main__":
    print_tool_call_accuracy_examples()
