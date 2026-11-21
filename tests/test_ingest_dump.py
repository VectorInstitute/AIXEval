"""ingest_dump turns one simple JSON object into a sealed record."""

from pathlib import Path

import pytest

from aixeval.ingest import ingest_dump
from aixeval.scorecard import build_profile, score_metrics


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_simple_match_scores_task_success() -> None:
    record = ingest_dump(FIXTURES / "simple_match.json")
    assert record == {
        "expected": "yes",
        "output": "yes",
        "stated_confidence": 1.0,
    }
    rows = score_metrics(record)
    by_id = {row["metric_id"]: row for row in rows}
    assert by_id["task_success"]["raw_score"] == 1.0
    assert by_id["verbalized_confidence"]["raw_score"] == 1.0
    assert by_id["grounding"]["status"] == "na"
    profile = build_profile(rows)
    assert profile["capability"] == 4
    assert profile["fairness"] is None


def test_letter_output_expands() -> None:
    record = ingest_dump(FIXTURES / "letter_choice.json")
    assert record == {"expected": "Paris", "output": "Paris"}
    assert "choices" not in record


def test_unknown_source_keys_are_dropped(tmp_path: Path) -> None:
    path = tmp_path / "alias.json"
    path.write_text('{"gold": "yes", "output": "yes"}\n', encoding="utf-8")
    record = ingest_dump(path)
    assert "gold" not in record
    assert "expected" not in record
    assert record["output"] == "yes"


def test_list_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "many.json"
    path.write_text("[{}]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="one JSON object"):
        ingest_dump(path)
