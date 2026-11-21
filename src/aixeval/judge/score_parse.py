"""Shared strict ``SCORE:`` line parser for LLM judge replies."""

from __future__ import annotations

import re


_SCORE_LINE = re.compile(r"SCORE:\s*([01](?:\.\d+)?|0?\.\d+)\s*$", re.IGNORECASE)


def parse_score(reply: str) -> float:
    """Return the last ``SCORE: <number>`` line's value. Strict: no guessing.

    Parameters
    ----------
    reply
        Full judge model reply text.

    Returns
    -------
    float
        The parsed score, in ``[0, 1]``.

    Raises
    ------
    ValueError
        The reply had no parseable, in-range ``SCORE:`` line.
    """
    match = None
    for line in reply.strip().splitlines():
        found = _SCORE_LINE.match(line.strip())
        if found is not None:
            match = found
    if match is None:
        msg = f"No parseable SCORE: line in judge reply: {reply!r}"
        raise ValueError(msg)
    score = float(match.group(1))
    if not 0.0 <= score <= 1.0:
        msg = f"SCORE out of [0, 1]: {score}"
        raise ValueError(msg)
    return score
