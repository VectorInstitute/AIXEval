"""Step stream + confirm-gate. Empty until the posthoc bar ships."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def watch_steps(steps: Iterator[dict[str, Any]]) -> list[dict[str, Any]]:
    """Watch a live step stream. Not this milestone."""
    del steps
    msg = "live harness is Milestone 2"
    raise NotImplementedError(msg)
