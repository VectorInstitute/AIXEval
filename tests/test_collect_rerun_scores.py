"""collect_rerun_scores: call a runner n times, collect one field's score each time."""

import functools

from aixeval.judge.runner import collect_rerun_scores, run_abstention_judge


class _FakeClient:
    """Returns replies from a fixed list, one per call, in order."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self.calls = 0

    @property
    def model(self) -> str:
        return "fake-model"

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        reply = self._replies[self.calls]
        self.calls += 1
        return reply


def _fake_run_once(results: list[tuple[dict, dict]]) -> callable:
    """Return a zero-arg callable yielding one result per call, in order."""
    iterator = iter(results)
    return lambda: next(iterator)


def test_default_n_is_three() -> None:
    run_once = _fake_run_once(
        [
            ({"x": 0.1}, {"status": "scored"}),
            ({"x": 0.2}, {"status": "scored"}),
            ({"x": 0.3}, {"status": "scored"}),
        ]
    )
    assert collect_rerun_scores(run_once, "x") == [0.1, 0.2, 0.3]


def test_n_is_configurable() -> None:
    run_once = _fake_run_once(
        [
            ({"x": 0.1}, {"status": "scored"}),
            ({"x": 0.2}, {"status": "scored"}),
        ]
    )
    assert collect_rerun_scores(run_once, "x", n=2) == [0.1, 0.2]


def test_failed_calls_are_excluded_not_counted_as_zero() -> None:
    run_once = _fake_run_once(
        [
            ({"x": 0.1}, {"status": "scored"}),
            ({}, {"status": "failed", "reason": "judge call failed"}),
            ({"x": 0.3}, {"status": "scored"}),
        ]
    )
    assert collect_rerun_scores(run_once, "x") == [0.1, 0.3]


def test_missing_or_non_numeric_field_is_excluded() -> None:
    run_once = _fake_run_once(
        [
            ({}, {"status": "scored"}),  # field missing even though call "succeeded"
            ({"x": "not a number"}, {"status": "scored"}),
            ({"x": True}, {"status": "scored"}),  # bool explicitly excluded, same as every other validator
            ({"x": 0.5}, {"status": "scored"}),
        ]
    )
    assert collect_rerun_scores(run_once, "x", n=4) == [0.5]


def test_all_calls_failing_returns_an_empty_list() -> None:
    run_once = _fake_run_once([({}, {"status": "failed"})] * 3)
    assert collect_rerun_scores(run_once, "x") == []


def test_composes_with_a_real_runner_via_partial() -> None:
    record = {"should_abstain": True, "output": "I don't have enough information."}
    client = _FakeClient(["SCORE: 0.8", "SCORE: 0.85", "SCORE: 0.5"])
    run_once = functools.partial(run_abstention_judge, record, client)

    scores = collect_rerun_scores(run_once, "abstention_judge")

    assert scores == [0.8, 0.85, 0.5]
    assert client.calls == 3


def test_each_call_is_independent_not_cached() -> None:
    # Three real calls happen even though the input never changes; that's
    # the whole point of a test-retest check, not an optimization to make.
    record = {"should_abstain": True, "output": "I don't know."}
    client = _FakeClient(["SCORE: 1", "SCORE: 1", "SCORE: 1"])
    run_once = functools.partial(run_abstention_judge, record, client)

    collect_rerun_scores(run_once, "abstention_judge", n=3)

    assert client.calls == 3
