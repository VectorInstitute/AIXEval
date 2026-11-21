"""Each judge wires client replies through parse_score and passes max_tokens."""

from collections.abc import Callable
from typing import Any

import pytest

from aixeval.judge.abstention import judge_abstention
from aixeval.judge.cross_modal_hallucination import judge_cross_modal_hallucination
from aixeval.judge.grounding import judge_grounding
from aixeval.judge.pairwise_preference import judge_pairwise_preference
from aixeval.judge.protected_attribute_bias import judge_protected_attribute_bias
from aixeval.judge.refusal import judge_refusal
from aixeval.judge.tool_call_accuracy import judge_tool_call_accuracy
from aixeval.judge.toxicity import judge_toxicity
from aixeval.judge.trajectory_optimality import judge_trajectory_optimality


class _FakeClient:
    def __init__(self, reply: str | list[str]) -> None:
        self._replies = [reply] if isinstance(reply, str) else list(reply)
        self.calls: list[dict[str, object]] = []

    @property
    def model(self) -> str:
        return "fake-model"

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        reply = self._replies[len(self.calls)]
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens})
        return reply

    def complete_with_image(self, prompt: str, image_path: str, *, max_tokens: int = 1024) -> str:
        reply = self._replies[len(self.calls)]
        self.calls.append({"prompt": prompt, "image_path": image_path, "max_tokens": max_tokens})
        return reply


def _call_abstention(client: _FakeClient, **kwargs: Any) -> float:
    return judge_abstention("output", client, **kwargs)


def _call_refusal(client: _FakeClient, **kwargs: Any) -> float:
    return judge_refusal("output", client, **kwargs)


def _call_toxicity(client: _FakeClient, **kwargs: Any) -> float:
    return judge_toxicity("output", client, **kwargs)


def _call_grounding(client: _FakeClient, **kwargs: Any) -> float:
    return judge_grounding("evidence", "output", client, **kwargs)


def _call_trajectory(client: _FakeClient, **kwargs: Any) -> float:
    return judge_trajectory_optimality("trajectory", "output", client, **kwargs)


def _call_tool_call(client: _FakeClient, **kwargs: Any) -> float:
    return judge_tool_call_accuracy("trajectory", "gold plan", "output", client, **kwargs)


def _call_cross_modal(client: _FakeClient, **kwargs: Any) -> float:
    return judge_cross_modal_hallucination("output", client, image_description="description", **kwargs)


def _call_bias(client: _FakeClient, **kwargs: Any) -> float:
    return judge_protected_attribute_bias("signal", "output", client, **kwargs)


def _call_pairwise(client: _FakeClient, **kwargs: Any) -> float:
    return judge_pairwise_preference("output", "baseline", client, **kwargs)


_JudgeCall = Callable[..., float]

_SINGLE_CALL_JUDGES: list[tuple[str, _JudgeCall]] = [
    ("abstention", _call_abstention),
    ("refusal", _call_refusal),
    ("toxicity", _call_toxicity),
    ("grounding", _call_grounding),
    ("trajectory_optimality", _call_trajectory),
    ("tool_call_accuracy", _call_tool_call),
    ("cross_modal_hallucination", _call_cross_modal),
    ("protected_attribute_bias", _call_bias),
]
_SINGLE_CALL_IDS = [name for name, _ in _SINGLE_CALL_JUDGES]
_SINGLE_CALL_FUNCS = [call for _, call in _SINGLE_CALL_JUDGES]


@pytest.mark.parametrize("call", _SINGLE_CALL_FUNCS, ids=_SINGLE_CALL_IDS)
def test_single_call_judge_returns_parsed_score(call: _JudgeCall) -> None:
    client = _FakeClient("Draft: SCORE: 0.2\nOn reflection:\nSCORE: 0.6")
    assert call(client) == 0.6
    assert len(client.calls) == 1


@pytest.mark.parametrize("call", _SINGLE_CALL_FUNCS, ids=_SINGLE_CALL_IDS)
def test_single_call_judge_default_max_tokens(call: _JudgeCall) -> None:
    client = _FakeClient("SCORE: 1")
    call(client)
    assert client.calls[0]["max_tokens"] == 4000


@pytest.mark.parametrize("call", _SINGLE_CALL_FUNCS, ids=_SINGLE_CALL_IDS)
def test_single_call_judge_max_tokens_is_overridable(call: _JudgeCall) -> None:
    client = _FakeClient("SCORE: 1")
    call(client, max_tokens=64)
    assert client.calls[0]["max_tokens"] == 64


def test_pairwise_returns_averaged_parsed_scores() -> None:
    client = _FakeClient(["SCORE: 1", "SCORE: 0"])
    assert _call_pairwise(client) == 1.0
    assert len(client.calls) == 2


def test_pairwise_default_max_tokens() -> None:
    client = _FakeClient(["SCORE: 1", "SCORE: 0"])
    _call_pairwise(client)
    assert client.calls[0]["max_tokens"] == 4000
    assert client.calls[1]["max_tokens"] == 4000


def test_pairwise_max_tokens_is_overridable() -> None:
    client = _FakeClient(["SCORE: 1", "SCORE: 0"])
    _call_pairwise(client, max_tokens=64)
    assert client.calls[0]["max_tokens"] == 64
    assert client.calls[1]["max_tokens"] == 64
