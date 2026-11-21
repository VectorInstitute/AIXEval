"""Closed N/A reasons. Starter set; add more ids here when a metric needs them."""

# Not a complete catalog. Scoring may only use an id from this tuple.
# Append a new snake_case id when a new "cannot score" case appears.
# Add the sentence in definitions.py in the same change.
NA_REASONS: tuple[str, ...] = (
    "no_gold",
    "missing_output",
    "missing_evidence",
    "cancelled",
    "not_in_scope",
)


def is_na_reason(reason: str) -> bool:
    """Return True if ``reason`` is in the current closed list."""
    return reason in NA_REASONS


def require_na_reason(reason: str) -> str:
    """Return ``reason`` if it is listed. Unknown ids are rejected."""
    if not is_na_reason(reason):
        msg = f"Unknown N/A reason {reason!r}. Add it to NA_REASONS if it is new."
        raise ValueError(msg)
    return reason
