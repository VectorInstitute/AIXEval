"""judge_pairwise_preference: two-call position-bias correction.

Parse / default max_tokens live in shared suites.
"""

import pytest

from aixeval.judge.pairwise_preference import judge_pairwise_preference


class _FakeClient:
    """Returns replies from a fixed list, one per call, in order."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self.calls: list[dict[str, object]] = []

    @property
    def model(self) -> str:
        return "fake-model"

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        reply = self._replies[len(self.calls)]
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens})
        return reply


def test_makes_exactly_two_calls() -> None:
    client = _FakeClient(["SCORE: 1", "SCORE: 0"])
    judge_pairwise_preference("output", "baseline", client)
    assert len(client.calls) == 2


def test_pure_position_bias_is_averaged_to_a_tie() -> None:
    """A judge that always prefers whichever response is shown first cancels out to 0.5.

    forward: output shown first -> "A" (output) wins -> 1.0.
    backward: baseline shown first -> "A" (baseline) wins -> 1.0, so output
    (shown second, as "B") is implied to lose that round -> 1 - 1.0 = 0.0.
    Averaging (1.0 + 0.0) / 2 = 0.5 is exactly the point: a judge with pure
    position bias and no real preference should net out to a tie, not a win.
    """
    client = _FakeClient(["SCORE: 1", "SCORE: 1"])
    assert judge_pairwise_preference("output", "baseline", client) == 0.5


def test_output_wins_both_presentation_orders() -> None:
    # forward: output=A wins (1.0). backward: baseline=A loses (0.0), so
    # output (shown as B) is implied to win -> 1 - 0.0 = 1.0. Average: 1.0.
    client = _FakeClient(["SCORE: 1", "SCORE: 0"])
    assert judge_pairwise_preference("output", "baseline", client) == 1.0


def test_baseline_wins_both_presentation_orders() -> None:
    # forward: output=A loses (0.0). backward: baseline=A wins (1.0), so
    # output (shown as B) is implied to lose -> 1 - 1.0 = 0.0. Average: 0.0.
    client = _FakeClient(["SCORE: 0", "SCORE: 1"])
    assert judge_pairwise_preference("output", "baseline", client) == 0.0


def test_reasoning_before_the_score_line() -> None:
    client = _FakeClient(["Response A is more complete.\nSCORE: 0.8", "Response B is more complete.\nSCORE: 0.2"])
    score = judge_pairwise_preference("output", "baseline", client)
    assert score == pytest.approx(0.8)


def test_sends_both_responses_in_each_prompt() -> None:
    client = _FakeClient(["SCORE: 1", "SCORE: 0"])
    judge_pairwise_preference("the specific output text", "the specific baseline text", client)
    forward_prompt = client.calls[0]["prompt"]
    backward_prompt = client.calls[1]["prompt"]
    assert "the specific output text" in forward_prompt
    assert "the specific baseline text" in forward_prompt
    assert "the specific output text" in backward_prompt
    assert "the specific baseline text" in backward_prompt


def test_reference_included_when_given() -> None:
    client = _FakeClient(["SCORE: 1", "SCORE: 0"])
    judge_pairwise_preference("output", "baseline", client, reference="the gold answer")
    assert "the gold answer" in client.calls[0]["prompt"]
    assert "the gold answer" in client.calls[1]["prompt"]


def test_no_reference_by_default() -> None:
    client = _FakeClient(["SCORE: 1", "SCORE: 0"])
    judge_pairwise_preference("output", "baseline", client)
    assert "REFERENCE ANSWER" not in client.calls[0]["prompt"]
