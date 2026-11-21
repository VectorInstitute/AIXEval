"""N/A uses a listed reason. The starter list is not the full catalog."""

import pytest

from aixeval.scorecard import NA_REASONS, is_na_reason, require_na_reason


def test_starter_reasons_include_the_common_gaps() -> None:
    assert "no_gold" in NA_REASONS
    assert "missing_output" in NA_REASONS
    assert "missing_evidence" in NA_REASONS


def test_reasons_are_unique_snake_case() -> None:
    assert NA_REASONS
    assert len(NA_REASONS) == len(set(NA_REASONS))
    for reason in NA_REASONS:
        assert reason == reason.lower()
        assert " " not in reason


def test_unknown_reason_is_rejected() -> None:
    assert is_na_reason("no_gold")
    assert not is_na_reason("silent_zero")
    with pytest.raises(ValueError, match="Unknown N/A reason"):
        require_na_reason("silent_zero")
    assert require_na_reason("cancelled") == "cancelled"


def test_starter_list_is_open_to_more_ids() -> None:
    """The tuple is closed at score time, not frozen as a product catalog."""
    assert NA_REASONS == (
        "no_gold",
        "missing_output",
        "missing_evidence",
        "cancelled",
        "not_in_scope",
    )
