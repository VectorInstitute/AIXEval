"""error_recovery is 1.0, 0.0, or N/A, then a 0-4 grade when scored."""

from aixeval.scorecard import score_error_recovery


def test_recovered_after_failure_is_one() -> None:
    result = score_error_recovery({"had_failure": True, "recovered": True})
    assert result == {
        "metric_id": "error_recovery",
        "status": "scored",
        "raw_score": 1.0,
        "grade": 4,
    }


def test_stuck_after_failure_is_zero() -> None:
    result = score_error_recovery({"had_failure": True, "recovered": False})
    assert result["status"] == "scored"
    assert result["raw_score"] == 0.0
    assert result["grade"] == 0


def test_no_failure_is_not_in_scope() -> None:
    result = score_error_recovery({"had_failure": False})
    assert result == {
        "metric_id": "error_recovery",
        "status": "na",
        "na_reason": "not_in_scope",
    }
    assert "raw_score" not in result
    assert "grade" not in result


def test_missing_flags_are_na_not_zero() -> None:
    result = score_error_recovery({"output": "yes"})
    assert result == {
        "metric_id": "error_recovery",
        "status": "na",
        "na_reason": "missing_evidence",
    }


def test_failure_without_recovered_is_missing_evidence() -> None:
    result = score_error_recovery({"had_failure": True})
    assert result["na_reason"] == "missing_evidence"


def test_non_bool_flags_are_missing_evidence() -> None:
    assert score_error_recovery({"had_failure": 1, "recovered": True})["na_reason"] == "missing_evidence"
    assert score_error_recovery({"had_failure": True, "recovered": 0})["na_reason"] == "missing_evidence"


def test_cancelled_beats_a_recovery() -> None:
    result = score_error_recovery({"had_failure": True, "recovered": True, "cancelled": True})
    assert result["status"] == "na"
    assert result["na_reason"] == "cancelled"
