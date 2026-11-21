"""The scorecard uses five grades, 0 through 4."""

import pytest

from aixeval.scorecard import BAND_LABELS, BAND_MAX, BAND_MIN, label_for_band


def test_scale_is_five_grades_zero_through_four() -> None:
    assert BAND_MIN == 0
    assert BAND_MAX == 4
    assert len(BAND_LABELS) == 5
    assert BAND_LABELS[0] == "Unacceptable"
    assert BAND_LABELS[4] == "Excellent"


def test_label_for_each_band() -> None:
    assert label_for_band(2) == "Adequate"
    with pytest.raises(ValueError, match="outside"):
        label_for_band(10)
    with pytest.raises(ValueError, match="outside"):
        label_for_band(-1)
