"""test_retest_stability is pvariance(rerun_scores) / 0.10, clamped, or N/A."""

from aixeval.scorecard import build_profile, score_metrics, score_test_retest_stability


def test_perfect_agreement_is_zero_and_grade_four() -> None:
    result = score_test_retest_stability({"rerun_scores": [0.8, 0.8, 0.8]})
    assert result == {
        "metric_id": "test_retest_stability",
        "status": "scored",
        "raw_score": 0.0,
        "grade": 4,
    }


def test_small_spread_scores_low_variance() -> None:
    result = score_test_retest_stability({"rerun_scores": [0.8, 0.85, 0.82]})
    assert result["status"] == "scored"
    assert 0.0 < result["raw_score"] < 0.1
    assert result["grade"] == 4


def test_wide_spread_clips_to_one() -> None:
    # pvariance([0.9, 0.5, 0.1]) = 0.1067 > the 0.10 budget -> clipped to 1.0
    result = score_test_retest_stability({"rerun_scores": [0.9, 0.5, 0.1]})
    assert result["raw_score"] == 1.0
    assert result["grade"] == 0


def test_two_reruns_is_the_minimum_allowed() -> None:
    result = score_test_retest_stability({"rerun_scores": [0.6, 0.6]})
    assert result["status"] == "scored"
    assert result["raw_score"] == 0.0


def test_single_rerun_is_na() -> None:
    assert score_test_retest_stability({"rerun_scores": [0.8]})["na_reason"] == "missing_evidence"


def test_missing_field_is_na() -> None:
    assert score_test_retest_stability({})["na_reason"] == "missing_evidence"


def test_empty_list_is_na() -> None:
    assert score_test_retest_stability({"rerun_scores": []})["na_reason"] == "missing_evidence"


def test_out_of_range_score_is_na() -> None:
    assert score_test_retest_stability({"rerun_scores": [0.8, 1.5]})["na_reason"] == "missing_evidence"


def test_bool_in_list_is_na() -> None:
    assert score_test_retest_stability({"rerun_scores": [0.8, True]})["na_reason"] == "missing_evidence"


def test_non_list_is_na() -> None:
    assert score_test_retest_stability({"rerun_scores": 0.8})["na_reason"] == "missing_evidence"


def test_cancelled_beats_perfect_agreement() -> None:
    result = score_test_retest_stability({"rerun_scores": [0.8, 0.8], "cancelled": True})
    assert result["na_reason"] == "cancelled"


def test_fills_robustness_on_the_card() -> None:
    profile = build_profile(
        score_metrics(
            {
                "expected": "yes",
                "output": "yes",
                "grounding_judge": 1.0,
                "rerun_scores": [0.9, 0.5, 0.1],
            }
        )
    )
    assert profile["robustness"] == 0


def test_distinct_from_consistency_on_the_same_record() -> None:
    # consistency (model behavior across two runs) and test_retest_stability (judge
    # agreement on a rerun) can disagree on the same record: different constructs.
    scored = score_metrics(
        {
            "expected": "yes",
            "output": "yes",
            "run_a_score": 0.8,
            "run_b_score": 0.8,  # consistency: perfect agreement
            "rerun_scores": [0.9, 0.5, 0.1],  # test_retest_stability: wide spread
        }
    )
    by_id = {row["metric_id"]: row for row in scored}
    assert by_id["consistency"]["grade"] == 4
    assert by_id["test_retest_stability"]["grade"] == 0
