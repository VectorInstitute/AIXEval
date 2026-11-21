"""Stubs exist and stay unimplemented."""

from __future__ import annotations

import pytest

from aixeval.harness import watch_steps
from aixeval.interpretability import attach_explanations
from aixeval.scorecard import DIMENSIONS, HARD_STOPS
from aixeval.serve import serve


def test_eight_dimensions() -> None:
    assert len(DIMENSIONS) == 8
    assert "capability" in DIMENSIONS
    assert len(HARD_STOPS) == 3


@pytest.mark.parametrize(
    "fn",
    [
        lambda: attach_explanations({}),
        lambda: watch_steps(iter(())),
        serve,
    ],
)
def test_public_entrypoints_are_stubs(fn: object) -> None:
    with pytest.raises(NotImplementedError):
        fn()  # type: ignore[operator]
