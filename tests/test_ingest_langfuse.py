"""ingest_langfuse copies declared top-level paths only."""

from pathlib import Path

import pytest

from aixeval.ingest import ingest_langfuse
from aixeval.scorecard import score_metrics


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_nested_gist_copies_declared_fields_only() -> None:
    record = ingest_langfuse(FIXTURES / "langfuse_gist.json")
    assert record == {
        "expected": "Paris",
        "output": "Paris",
        "stated_confidence": 0.9,
        "latency_sec": 3.2,
        "steps": 2,
    }
    rows = score_metrics(record)
    by_id = {row["metric_id"]: row for row in rows}
    assert by_id["task_success"]["raw_score"] == 1.0
    assert by_id["latency_vs_budget"]["status"] == "na"
    assert by_id["step_efficiency"]["status"] == "na"


def test_root_output_wins_over_generation() -> None:
    record = ingest_langfuse(FIXTURES / "langfuse_gist.json")
    assert record["output"] == "Paris"
    assert record["output"] != "planning"


def test_last_generation_output_when_root_has_none() -> None:
    record = ingest_langfuse(FIXTURES / "langfuse_generation_only.json")
    assert record["output"] == "Paris"
    assert record["expected"] == "Paris"
    assert record["steps"] == 3
    rows = score_metrics(record)
    by_id = {row["metric_id"]: row for row in rows}
    assert by_id["task_success"]["raw_score"] == 1.0


def test_observations_only_does_not_infer_gold_output_or_error() -> None:
    record = ingest_langfuse(FIXTURES / "langfuse_no_gold.json")
    assert record == {"steps": 2}
    rows = score_metrics(record)
    by_id = {row["metric_id"]: row for row in rows}
    assert by_id["task_success"]["na_reason"] == "no_gold"
    assert by_id["error_recovery"]["na_reason"] == "missing_evidence"
    assert by_id["latency_vs_budget"]["na_reason"] == "missing_evidence"


def test_list_export_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "events.json"
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="one JSON object"):
        ingest_langfuse(path)
