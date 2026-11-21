"""Behavioral / logit / mechanistic notes. Not a ninth dimension."""

from __future__ import annotations

from typing import Any


def attach_explanations(record: dict[str, Any]) -> dict[str, Any]:
    """Return an attachment. Mechanistic methods N/A without weights."""
    del record
    msg = "attach_explanations is not implemented"
    raise NotImplementedError(msg)
