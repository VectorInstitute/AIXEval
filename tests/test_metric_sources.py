"""Every starter metric names a literature or policy source."""

from aixeval.scorecard.metrics import POSTHOC_METRICS
from aixeval.scorecard.sources import (
    DIMENSION_FRAME,
    METRIC_SOURCES,
    assert_every_starter_metric_has_a_source,
)


def test_every_starter_metric_has_a_source() -> None:
    assert_every_starter_metric_has_a_source()
    assert set(METRIC_SOURCES) == set(POSTHOC_METRICS)


def test_sources_are_literature_or_policy() -> None:
    for row in METRIC_SOURCES.values():
        assert row["kind"] in {"literature", "policy"}
        assert row["citation"]
        assert row["note"]


def test_dimension_frame_is_helm_and_nist_not_a_new_science() -> None:
    assert DIMENSION_FRAME["kind"] == "literature"
    assert "HELM" in DIMENSION_FRAME["citation"]
    assert "NIST" in DIMENSION_FRAME["citation"]
