"""Every named id has a one-sentence definition in the package."""

import pytest

from aixeval.scorecard import (
    ALL_METRIC_IDS,
    DIMENSION_DEFINITIONS,
    DIMENSIONS,
    HARD_STOP_DEFINITIONS,
    HARD_STOPS,
    METRIC_DEFINITIONS,
    NA_REASON_DEFINITIONS,
    NA_REASONS,
    OPT_IN_METRIC_DEFINITIONS,
    OPT_IN_METRICS,
    POSTHOC_METRICS,
    definition_for_any_metric,
    definition_for_metric,
    definition_for_na_reason,
)


def test_every_starter_metric_has_a_definition() -> None:
    assert set(METRIC_DEFINITIONS) == set(POSTHOC_METRICS)
    for metric_id in POSTHOC_METRICS:
        text = definition_for_metric(metric_id)
        assert text
        assert text == METRIC_DEFINITIONS[metric_id]


def test_every_na_reason_has_a_definition() -> None:
    assert set(NA_REASON_DEFINITIONS) == set(NA_REASONS)
    for reason in NA_REASONS:
        text = definition_for_na_reason(reason)
        assert text
        assert text == NA_REASON_DEFINITIONS[reason]


def test_every_dimension_and_hard_stop_has_a_definition() -> None:
    assert set(DIMENSION_DEFINITIONS) == set(DIMENSIONS)
    assert set(HARD_STOP_DEFINITIONS) == set(HARD_STOPS)
    for text in (*DIMENSION_DEFINITIONS.values(), *HARD_STOP_DEFINITIONS.values()):
        assert text


def test_unknown_metric_has_no_definition() -> None:
    with pytest.raises(ValueError, match="Unknown metric"):
        definition_for_metric("not_a_metric")


def test_every_opt_in_metric_has_a_definition() -> None:
    assert set(OPT_IN_METRIC_DEFINITIONS) == set(OPT_IN_METRICS)
    for metric_id in OPT_IN_METRICS:
        assert definition_for_any_metric(metric_id) == OPT_IN_METRIC_DEFINITIONS[metric_id]


def test_all_metric_ids_is_posthoc_plus_opt_in_no_overlap() -> None:
    assert set(POSTHOC_METRICS).isdisjoint(OPT_IN_METRICS)
    assert (*POSTHOC_METRICS, *OPT_IN_METRICS) == ALL_METRIC_IDS
    for metric_id in ALL_METRIC_IDS:
        assert definition_for_any_metric(metric_id)


def test_definition_for_any_metric_rejects_unknown_id() -> None:
    with pytest.raises(ValueError, match="Unknown metric"):
        definition_for_any_metric("not_a_metric")
