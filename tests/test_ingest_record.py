"""validate_record allows missing keys and rejects bad names or types."""

import pytest

from aixeval.ingest import validate_record


def test_empty_record_is_valid() -> None:
    assert validate_record({}) == {}


def test_known_fields_pass() -> None:
    record = validate_record({"expected": "yes", "output": "yes", "stated_confidence": 0.5})
    assert record["stated_confidence"] == 0.5


def test_unknown_key_fails() -> None:
    with pytest.raises(ValueError, match="Unknown sealed-record field"):
        validate_record({"gold": "yes"})


def test_bad_confidence_fails() -> None:
    with pytest.raises(ValueError, match="stated_confidence"):
        validate_record({"stated_confidence": 1.5})
    with pytest.raises(ValueError, match="stated_confidence"):
        validate_record({"stated_confidence": True})


def test_bool_must_be_bool() -> None:
    with pytest.raises(ValueError, match="cancelled"):
        validate_record({"cancelled": 1})


def test_gold_tool_calls_passes_through() -> None:
    calls = [{"tool": "get_weather", "args": {"city": "Paris"}}]
    record = validate_record({"gold_tool_calls": calls})
    assert record["gold_tool_calls"] == calls


def test_gold_tool_calls_args_defaults_to_empty_dict() -> None:
    record = validate_record({"gold_tool_calls": [{"tool": "ping"}]})
    assert record["gold_tool_calls"] == [{"tool": "ping", "args": {}}]


def test_gold_tool_calls_must_be_a_list() -> None:
    with pytest.raises(ValueError, match="gold_tool_calls must be a list"):
        validate_record({"gold_tool_calls": "not a list"})


def test_gold_tool_calls_items_must_be_objects() -> None:
    with pytest.raises(ValueError, match="gold_tool_calls"):
        validate_record({"gold_tool_calls": ["not a dict"]})


def test_gold_tool_calls_items_need_a_non_empty_tool_name() -> None:
    with pytest.raises(ValueError, match="gold_tool_calls"):
        validate_record({"gold_tool_calls": [{"args": {}}]})
    with pytest.raises(ValueError, match="gold_tool_calls"):
        validate_record({"gold_tool_calls": [{"tool": ""}]})


def test_gold_tool_calls_args_must_be_an_object() -> None:
    with pytest.raises(ValueError, match="gold_tool_calls"):
        validate_record({"gold_tool_calls": [{"tool": "ping", "args": "not a dict"}]})


def test_gold_tool_plan_is_text() -> None:
    record = validate_record({"gold_tool_plan": "Should call get_weather."})
    assert record["gold_tool_plan"] == "Should call get_weather."


def test_tool_call_accuracy_judge_is_unit_interval() -> None:
    record = validate_record({"tool_call_accuracy_judge": 0.5})
    assert record["tool_call_accuracy_judge"] == 0.5
    with pytest.raises(ValueError, match="tool_call_accuracy_judge"):
        validate_record({"tool_call_accuracy_judge": 1.5})


def test_cost_usd_and_budget_are_nonneg_floats() -> None:
    record = validate_record({"cost_usd": 0.003, "cost_budget_usd": 0.01})
    assert record["cost_usd"] == 0.003
    assert record["cost_budget_usd"] == 0.01
    with pytest.raises(ValueError, match="cost_usd"):
        validate_record({"cost_usd": -0.001})
    with pytest.raises(ValueError, match="cost_budget_usd"):
        validate_record({"cost_budget_usd": True})


def test_toxicity_judge_is_unit_interval() -> None:
    record = validate_record({"toxicity_judge": 0.5})
    assert record["toxicity_judge"] == 0.5
    with pytest.raises(ValueError, match="toxicity_judge"):
        validate_record({"toxicity_judge": 1.5})


def test_energy_kwh_and_budget_are_nonneg_floats() -> None:
    record = validate_record({"energy_kwh": 0.0003, "energy_budget_kwh": 0.001})
    assert record["energy_kwh"] == 0.0003
    assert record["energy_budget_kwh"] == 0.001
    with pytest.raises(ValueError, match="energy_kwh"):
        validate_record({"energy_kwh": -0.0001})
    with pytest.raises(ValueError, match="energy_budget_kwh"):
        validate_record({"energy_budget_kwh": True})


def test_oversight_bool_flags_pass() -> None:
    record = validate_record(
        {
            "stop_signal_sent": True,
            "halted_cleanly": False,
            "override_requested": True,
            "override_applied": False,
            "low_confidence_flagged": True,
            "deferred_to_human": False,
        }
    )
    assert record["stop_signal_sent"] is True
    assert record["halted_cleanly"] is False
    assert record["override_requested"] is True
    assert record["override_applied"] is False
    assert record["low_confidence_flagged"] is True
    assert record["deferred_to_human"] is False
    with pytest.raises(ValueError, match="stop_signal_sent"):
        validate_record({"stop_signal_sent": 1})
    with pytest.raises(ValueError, match="override_applied"):
        validate_record({"override_applied": "yes"})


def test_baseline_output_is_text() -> None:
    record = validate_record({"baseline_output": "The capital of France is Paris."})
    assert record["baseline_output"] == "The capital of France is Paris."


def test_pairwise_preference_judge_is_unit_interval() -> None:
    record = validate_record({"pairwise_preference_judge": 0.5})
    assert record["pairwise_preference_judge"] == 0.5
    with pytest.raises(ValueError, match="pairwise_preference_judge"):
        validate_record({"pairwise_preference_judge": 1.5})


def test_retrieval_rank_and_k_are_nonneg_ints() -> None:
    record = validate_record({"retrieval_rank": 1, "retrieval_k": 5})
    assert record["retrieval_rank"] == 1
    assert record["retrieval_k"] == 5
    with pytest.raises(ValueError, match="retrieval_rank"):
        validate_record({"retrieval_rank": -1})
    with pytest.raises(ValueError, match="retrieval_k"):
        validate_record({"retrieval_k": True})


def test_bbox_fields_pass() -> None:
    record = validate_record({"predicted_bbox": [1, 2, 3, 4], "gold_bbox": [1.0, 2.0, 3.0, 4.0]})
    assert record["predicted_bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert record["gold_bbox"] == [1.0, 2.0, 3.0, 4.0]


def test_bbox_must_have_exactly_four_numbers() -> None:
    with pytest.raises(ValueError, match="predicted_bbox"):
        validate_record({"predicted_bbox": [1, 2, 3]})
    with pytest.raises(ValueError, match="gold_bbox"):
        validate_record({"gold_bbox": "not a box"})


def test_clean_and_perturbed_score_are_unit_interval() -> None:
    record = validate_record({"clean_score": 0.9, "perturbed_score": 0.4})
    assert record["clean_score"] == 0.9
    assert record["perturbed_score"] == 0.4
    with pytest.raises(ValueError, match="clean_score"):
        validate_record({"clean_score": 1.5})


def test_gold_answers_and_reference_captions_pass() -> None:
    record = validate_record({"gold_answers": ["yellow", "gold"], "reference_captions": ["a dog on a beach"]})
    assert record["gold_answers"] == ["yellow", "gold"]
    assert record["reference_captions"] == ["a dog on a beach"]


def test_str_list_items_must_be_non_empty_strings() -> None:
    with pytest.raises(ValueError, match="gold_answers"):
        validate_record({"gold_answers": ["yellow", ""]})
    with pytest.raises(ValueError, match="reference_captions"):
        validate_record({"reference_captions": [1, 2]})
    with pytest.raises(ValueError, match="gold_answers"):
        validate_record({"gold_answers": "not a list"})


def test_image_description_and_image_path_are_text() -> None:
    record = validate_record({"image_description": "A dog on a dock.", "image_path": "/data/dog.jpg"})
    assert record["image_description"] == "A dog on a dock."
    assert record["image_path"] == "/data/dog.jpg"


def test_cross_modal_hallucination_judge_is_unit_interval() -> None:
    record = validate_record({"cross_modal_hallucination_judge": 0.5})
    assert record["cross_modal_hallucination_judge"] == 0.5
    with pytest.raises(ValueError, match="cross_modal_hallucination_judge"):
        validate_record({"cross_modal_hallucination_judge": 1.5})


def test_subgoals_completed_and_total_are_nonneg_ints() -> None:
    record = validate_record({"subgoals_completed": 1, "subgoals_total": 4})
    assert record["subgoals_completed"] == 1
    assert record["subgoals_total"] == 4
    with pytest.raises(ValueError, match="subgoals_completed"):
        validate_record({"subgoals_completed": -1})
    with pytest.raises(ValueError, match="subgoals_total"):
        validate_record({"subgoals_total": True})


def test_attempt_budget_is_nonneg_int_and_solved_within_budget_is_bool() -> None:
    record = validate_record({"attempt_budget": 3, "solved_within_budget": True})
    assert record["attempt_budget"] == 3
    assert record["solved_within_budget"] is True
    with pytest.raises(ValueError, match="attempt_budget"):
        validate_record({"attempt_budget": -1})
    with pytest.raises(ValueError, match="solved_within_budget"):
        validate_record({"solved_within_budget": "yes"})


def test_run_a_and_run_b_score_are_unit_interval() -> None:
    record = validate_record({"run_a_score": 0.9, "run_b_score": 0.4})
    assert record["run_a_score"] == 0.9
    assert record["run_b_score"] == 0.4
    with pytest.raises(ValueError, match="run_a_score"):
        validate_record({"run_a_score": 1.5})


def test_terminated_and_verified_flags_are_bools() -> None:
    record = validate_record({"terminated_correctly": True, "verified_correctly": False})
    assert record["terminated_correctly"] is True
    assert record["verified_correctly"] is False
    with pytest.raises(ValueError, match="terminated_correctly"):
        validate_record({"terminated_correctly": "yes"})


def test_judge_provenance_passes_through() -> None:
    provenance = {"grounding": {"model": "gemma-3-12b-it", "prompt_version": "grounding-v1"}}
    record = validate_record({"judge_provenance": provenance})
    assert record["judge_provenance"] == provenance


def test_judge_provenance_rejects_empty_or_incomplete_entries() -> None:
    with pytest.raises(ValueError, match="judge_provenance"):
        validate_record({"judge_provenance": {}})
    with pytest.raises(ValueError, match="judge_provenance"):
        validate_record({"judge_provenance": {"grounding": {"model": "x"}}})
    with pytest.raises(ValueError, match="judge_provenance"):
        validate_record({"judge_provenance": {"grounding": "not an object"}})


def test_group_outcomes_is_a_mapping_of_at_least_two_unit_rates() -> None:
    record = validate_record({"group_outcomes": {"a": 0.5, "b": 0.3}})
    assert record["group_outcomes"] == {"a": 0.5, "b": 0.3}
    with pytest.raises(ValueError, match="group_outcomes"):
        validate_record({"group_outcomes": {"a": 0.5}})
    with pytest.raises(ValueError, match="group_outcomes"):
        validate_record({"group_outcomes": [0.5, 0.3]})
    with pytest.raises(ValueError, match="group_outcomes"):
        validate_record({"group_outcomes": {"a": 1.5, "b": 0.3}})


def test_pass_hat_k_counts_are_nonneg_ints() -> None:
    record = validate_record({"pass_hat_k_attempts": 5, "pass_hat_k_successes": 4})
    assert record["pass_hat_k_attempts"] == 5
    assert record["pass_hat_k_successes"] == 4
    with pytest.raises(ValueError, match="pass_hat_k_attempts"):
        validate_record({"pass_hat_k_attempts": -1})
    with pytest.raises(ValueError, match="pass_hat_k_successes"):
        validate_record({"pass_hat_k_successes": True})
