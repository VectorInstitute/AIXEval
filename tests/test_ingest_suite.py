"""ingest_suite loads a JSON array or JSONL into many sealed records."""

from pathlib import Path

import pytest

from aixeval.ingest import ingest_dump, ingest_suite
from aixeval.scorecard import score_metrics


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _expected_suite() -> list[dict[str, object]]:
    return [
        {"expected": "yes", "output": "yes"},
        {"expected": "Paris", "output": "Paris"},
        {"output": "no gold on this row"},
    ]


def test_json_array_suite() -> None:
    records = ingest_suite(FIXTURES / "suite.json")
    assert records == _expected_suite()
    first = score_metrics(records[0])
    third = score_metrics(records[2])
    by_first = {row["metric_id"]: row for row in first}
    by_third = {row["metric_id"]: row for row in third}
    assert by_first["task_success"]["raw_score"] == 1.0
    assert by_third["task_success"]["na_reason"] == "no_gold"


def test_jsonl_suite() -> None:
    assert ingest_suite(FIXTURES / "suite.jsonl") == _expected_suite()


def test_single_object_is_not_a_suite() -> None:
    with pytest.raises(ValueError, match="JSON array or a .jsonl"):
        ingest_suite(FIXTURES / "simple_match.json")


def test_ingest_dump_still_rejects_an_array() -> None:
    with pytest.raises(ValueError, match="one JSON object"):
        ingest_dump(FIXTURES / "suite.json")


def test_bad_row_names_the_index(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('[{"expected": "yes"}, {"stated_confidence": 2}]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="suite item 1"):
        ingest_suite(path)
