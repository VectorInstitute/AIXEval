"""task_success is 1.0, 0.0, or N/A, then a 0-4 grade when scored."""

from aixeval.scorecard import score_task_success


def test_exact_match_is_one() -> None:
    result = score_task_success({"expected": "Paris", "output": "Paris"})
    assert result == {
        "metric_id": "task_success",
        "status": "scored",
        "raw_score": 1.0,
        "grade": 4,
    }


def test_mismatch_is_zero() -> None:
    result = score_task_success({"expected": "Paris", "output": "London"})
    assert result["status"] == "scored"
    assert result["raw_score"] == 0.0
    assert result["grade"] == 0


def test_strips_whitespace() -> None:
    result = score_task_success({"expected": "  yes  ", "output": "yes"})
    assert result["raw_score"] == 1.0


def test_veto_is_a_fail_not_na() -> None:
    """Veto means the harness blocked a high-impact action mid-run."""
    result = score_task_success({"expected": "yes", "output": "yes", "vetoed": True})
    assert result["status"] == "scored"
    assert result["raw_score"] == 0.0
    assert result["grade"] == 0


def test_no_gold_is_na_not_zero() -> None:
    result = score_task_success({"output": "Paris"})
    assert result == {"metric_id": "task_success", "status": "na", "na_reason": "no_gold"}
    assert "raw_score" not in result
    assert "grade" not in result


def test_missing_output_is_na_not_zero() -> None:
    result = score_task_success({"expected": "Paris"})
    assert result["na_reason"] == "missing_output"
    assert "raw_score" not in result


def test_empty_strings_are_missing() -> None:
    assert score_task_success({"expected": "  ", "output": "x"})["na_reason"] == "no_gold"
    assert score_task_success({"expected": "x", "output": ""})["na_reason"] == "missing_output"


def test_cancelled_beats_a_match() -> None:
    result = score_task_success({"expected": "yes", "output": "yes", "cancelled": True})
    assert result["status"] == "na"
    assert result["na_reason"] == "cancelled"


def test_wrong_field_type_is_missing_evidence() -> None:
    result = score_task_success({"expected": ["Paris"], "output": "Paris"})
    assert result["na_reason"] == "missing_evidence"
