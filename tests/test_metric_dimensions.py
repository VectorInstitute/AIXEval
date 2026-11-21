"""Starter metrics map onto the eight card questions. Names only."""

from aixeval.scorecard import (
    DIMENSIONS,
    METRIC_DIMENSION,
    POSTHOC_METRICS,
    metrics_for_dimension,
)


def test_every_starter_metric_has_a_known_dimension() -> None:
    assert set(METRIC_DIMENSION) == set(POSTHOC_METRICS)
    assert set(METRIC_DIMENSION.values()) <= set(DIMENSIONS)


def test_fairness_has_two_starter_metrics() -> None:
    assert metrics_for_dimension("fairness") == ("fairness_gap", "protected_attribute_bias")


def test_capability_and_efficiency_metrics() -> None:
    assert metrics_for_dimension("capability") == (
        "task_success",
        "grounding",
        "bleu",
        "rouge",
        "wer",
        "f1",
        "trajectory_optimality",
        "tool_call_accuracy",
        "pairwise_preference",
        "vqa_accuracy",
        "caption_quality",
        "cross_modal_retrieval",
        "grounding_accuracy",
        "ocr_text_read_accuracy",
        "sub_goal_progress",
        "pass_at_k_agent",
        "delegation_accuracy",
        "pass_hat_k_agent",
    )
    assert metrics_for_dimension("efficiency") == (
        "step_efficiency",
        "latency_vs_budget",
        "cost_per_query",
        "energy_per_query",
    )


def test_oversight_metrics() -> None:
    assert metrics_for_dimension("oversight") == (
        "confirmation_gating",
        "abstention",
        "interruptibility",
        "human_override_success",
        "deferral_on_uncertainty",
    )


def test_safety_metrics() -> None:
    assert metrics_for_dimension("safety") == (
        "unsafe_action",
        "refusal",
        "over_refusal",
        "cross_modal_hallucination",
        "cumulative_unsafe_action_mas",
    )


def test_robustness_metrics() -> None:
    assert metrics_for_dimension("robustness") == (
        "error_recovery",
        "modality_robustness",
        "consistency",
        "termination_correctness",
        "verification_correctness",
        "error_propagation_containment",
        "deadlock_livelock",
        "test_retest_stability",
    )


def test_trajectory_optimality_is_capability() -> None:
    assert "trajectory_optimality" in metrics_for_dimension("capability")
