"""Shared SCORE: line parser used by every LLM judge."""

import pytest

from aixeval.judge.score_parse import parse_score


def test_clean_score_line() -> None:
    assert parse_score("SCORE: 0.8") == 0.8


def test_reasoning_before_the_score_line() -> None:
    reply = "Let me think about this.\nThe output looks fine.\nSCORE: 1"
    assert parse_score(reply) == 1.0


def test_takes_the_last_score_line_if_several() -> None:
    reply = "Draft: SCORE: 0.2\nOn reflection:\nSCORE: 0.6"
    assert parse_score(reply) == 0.6


def test_integer_zero_and_one() -> None:
    assert parse_score("SCORE: 0") == 0.0
    assert parse_score("SCORE: 1") == 1.0


def test_case_insensitive_tag() -> None:
    assert parse_score("score: 0.5") == 0.5


def test_no_score_line_raises() -> None:
    with pytest.raises(ValueError, match="No parseable SCORE"):
        parse_score("I think it looks fine.")


def test_out_of_range_score_raises() -> None:
    with pytest.raises(ValueError, match=r"out of \[0, 1\]"):
        parse_score("SCORE: 1.5")


def test_negative_score_raises() -> None:
    with pytest.raises(ValueError, match="No parseable SCORE"):
        parse_score("SCORE: -0.1")
