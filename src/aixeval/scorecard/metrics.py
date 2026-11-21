"""Posthoc metric set: 43 auto-wired ids plus 4 opt-in (47 total)."""

from __future__ import annotations

from typing import Any


# bertscore and meteor are deliberately absent: both are tier-1 metrics
# too, but opt-in only. bertscore: torch/transformers + a multi-GB model
# download on first real call. meteor: its fragmentation-penalty formula
# scores short/single-token exact matches (e.g. "Paris"/"Paris") around
# 0.5, not ~1.0, since capability's card grade is the MIN across every
# scored capability metric, wiring meteor in would drag a perfect short
# answer's capability grade down even though task_success correctly
# grades it 4. Call score_bertscore / score_meteor directly; neither is
# added here and score_metrics() never invokes either automatically.
POSTHOC_METRICS: tuple[str, ...] = (
    "task_success",
    "contamination",
    "grounding",
    "step_efficiency",
    "error_recovery",
    "latency_vs_budget",
    "confirmation_gating",
    "unsafe_action",
    "verbalized_confidence",
    "bleu",
    "rouge",
    "wer",
    "f1",
    "abstention",
    "refusal",
    "over_refusal",
    "trajectory_optimality",
    "tool_call_accuracy",
    "cost_per_query",
    "energy_per_query",
    "interruptibility",
    "human_override_success",
    "deferral_on_uncertainty",
    "pairwise_preference",
    "vqa_accuracy",
    "caption_quality",
    "cross_modal_retrieval",
    "grounding_accuracy",
    "ocr_text_read_accuracy",
    "modality_robustness",
    "cross_modal_hallucination",
    "sub_goal_progress",
    "pass_at_k_agent",
    "consistency",
    "termination_correctness",
    "verification_correctness",
    "delegation_accuracy",
    "error_propagation_containment",
    "deadlock_livelock",
    "cumulative_unsafe_action_mas",
    "test_retest_stability",
    "fairness_gap",
    "protected_attribute_bias",
    "pass_hat_k_agent",
)

# Metrics that exist but are never called by score_metrics(); see the
# comment above POSTHOC_METRICS for why each one is opt-in. Callers
# invoke score_toxicity/score_bertscore/score_meteor/score_clip_score
# directly.
OPT_IN_METRICS: tuple[str, ...] = ("toxicity", "bertscore", "meteor", "clip_score")

# Every metric id this toolkit knows, auto-wired or not. For UI/suggester
# code that needs to present the full catalog, not just what
# score_metrics() computes automatically.
ALL_METRIC_IDS: tuple[str, ...] = POSTHOC_METRICS + OPT_IN_METRICS

# Starter set only. A question may have zero metrics (then it stays N/A).
METRIC_DIMENSION: dict[str, str] = {
    "task_success": "capability",
    "grounding": "capability",
    "bleu": "capability",
    "rouge": "capability",
    "wer": "capability",
    "f1": "capability",
    "trajectory_optimality": "capability",
    "tool_call_accuracy": "capability",
    "pairwise_preference": "capability",
    "vqa_accuracy": "capability",
    "caption_quality": "capability",
    "cross_modal_retrieval": "capability",
    "grounding_accuracy": "capability",
    "ocr_text_read_accuracy": "capability",
    "sub_goal_progress": "capability",
    "pass_at_k_agent": "capability",
    "pass_hat_k_agent": "capability",
    "error_recovery": "robustness",
    "modality_robustness": "robustness",
    "consistency": "robustness",
    "test_retest_stability": "robustness",
    "termination_correctness": "robustness",
    "verification_correctness": "robustness",
    "delegation_accuracy": "capability",
    "error_propagation_containment": "robustness",
    "deadlock_livelock": "robustness",
    "cumulative_unsafe_action_mas": "safety",
    "fairness_gap": "fairness",
    "protected_attribute_bias": "fairness",
    "unsafe_action": "safety",
    "refusal": "safety",
    "over_refusal": "safety",
    "cross_modal_hallucination": "safety",
    "verbalized_confidence": "transparency",
    "contamination": "governance",
    "confirmation_gating": "oversight",
    "abstention": "oversight",
    "interruptibility": "oversight",
    "human_override_success": "oversight",
    "deferral_on_uncertainty": "oversight",
    "latency_vs_budget": "efficiency",
    "step_efficiency": "efficiency",
    "cost_per_query": "efficiency",
    "energy_per_query": "efficiency",
}


def metrics_for_dimension(dimension: str) -> tuple[str, ...]:
    """Return starter metrics that inform one card question, in catalog order."""
    return tuple(metric_id for metric_id in POSTHOC_METRICS if METRIC_DIMENSION[metric_id] == dimension)


def score_metrics(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Score one sealed record against every id in ``POSTHOC_METRICS``.

    Fairness has two metrics, ``fairness_gap`` and
    ``protected_attribute_bias``; each stays N/A until its own evidence
    is declared. Missing fields are N/A with a reason, never a silent 0.
    Opt-in metrics (``OPT_IN_METRICS``) are not invoked here.
    """
    from aixeval.scorecard.abstention import score_abstention  # noqa: PLC0415
    from aixeval.scorecard.bleu import score_bleu  # noqa: PLC0415
    from aixeval.scorecard.caption_quality import score_caption_quality  # noqa: PLC0415
    from aixeval.scorecard.confirmation_gating import (  # noqa: PLC0415
        score_confirmation_gating,
    )
    from aixeval.scorecard.consistency import score_consistency  # noqa: PLC0415
    from aixeval.scorecard.contamination import score_contamination  # noqa: PLC0415
    from aixeval.scorecard.cost_per_query import score_cost_per_query  # noqa: PLC0415
    from aixeval.scorecard.cross_modal_hallucination import (  # noqa: PLC0415
        score_cross_modal_hallucination,
    )
    from aixeval.scorecard.cross_modal_retrieval import (  # noqa: PLC0415
        score_cross_modal_retrieval,
    )
    from aixeval.scorecard.cumulative_unsafe_action_mas import score_cumulative_unsafe_action_mas  # noqa: PLC0415
    from aixeval.scorecard.deadlock_livelock import score_deadlock_livelock  # noqa: PLC0415
    from aixeval.scorecard.deferral_on_uncertainty import (  # noqa: PLC0415
        score_deferral_on_uncertainty,
    )
    from aixeval.scorecard.delegation_accuracy import score_delegation_accuracy  # noqa: PLC0415
    from aixeval.scorecard.energy_per_query import score_energy_per_query  # noqa: PLC0415
    from aixeval.scorecard.error_propagation_containment import (  # noqa: PLC0415
        score_error_propagation_containment,
    )
    from aixeval.scorecard.error_recovery import score_error_recovery  # noqa: PLC0415
    from aixeval.scorecard.f1 import score_f1  # noqa: PLC0415
    from aixeval.scorecard.fairness_gap import score_fairness_gap  # noqa: PLC0415
    from aixeval.scorecard.grounding import score_grounding  # noqa: PLC0415
    from aixeval.scorecard.grounding_accuracy import score_grounding_accuracy  # noqa: PLC0415
    from aixeval.scorecard.human_override_success import (  # noqa: PLC0415
        score_human_override_success,
    )
    from aixeval.scorecard.interruptibility import score_interruptibility  # noqa: PLC0415
    from aixeval.scorecard.latency_vs_budget import (  # noqa: PLC0415
        score_latency_vs_budget,
    )
    from aixeval.scorecard.modality_robustness import score_modality_robustness  # noqa: PLC0415
    from aixeval.scorecard.ocr_text_read_accuracy import (  # noqa: PLC0415
        score_ocr_text_read_accuracy,
    )
    from aixeval.scorecard.over_refusal import score_over_refusal  # noqa: PLC0415
    from aixeval.scorecard.pairwise_preference import (  # noqa: PLC0415
        score_pairwise_preference,
    )
    from aixeval.scorecard.pass_at_k_agent import score_pass_at_k_agent  # noqa: PLC0415
    from aixeval.scorecard.pass_hat_k_agent import score_pass_hat_k_agent  # noqa: PLC0415
    from aixeval.scorecard.protected_attribute_bias import score_protected_attribute_bias  # noqa: PLC0415
    from aixeval.scorecard.refusal import score_refusal  # noqa: PLC0415
    from aixeval.scorecard.rouge import score_rouge  # noqa: PLC0415
    from aixeval.scorecard.step_efficiency import score_step_efficiency  # noqa: PLC0415
    from aixeval.scorecard.sub_goal_progress import score_sub_goal_progress  # noqa: PLC0415
    from aixeval.scorecard.task_success import score_task_success  # noqa: PLC0415
    from aixeval.scorecard.termination_correctness import score_termination_correctness  # noqa: PLC0415
    from aixeval.scorecard.test_retest_stability import score_test_retest_stability  # noqa: PLC0415
    from aixeval.scorecard.tool_call_accuracy import (  # noqa: PLC0415
        score_tool_call_accuracy,
    )
    from aixeval.scorecard.trajectory_optimality import (  # noqa: PLC0415
        score_trajectory_optimality,
    )
    from aixeval.scorecard.unsafe_action import score_unsafe_action  # noqa: PLC0415
    from aixeval.scorecard.verbalized_confidence import (  # noqa: PLC0415
        score_verbalized_confidence,
    )
    from aixeval.scorecard.verification_correctness import score_verification_correctness  # noqa: PLC0415
    from aixeval.scorecard.vqa_accuracy import score_vqa_accuracy  # noqa: PLC0415
    from aixeval.scorecard.wer import score_wer  # noqa: PLC0415

    return [
        score_task_success(record),
        score_grounding(record),
        score_trajectory_optimality(record),
        score_tool_call_accuracy(record),
        score_pairwise_preference(record),
        score_vqa_accuracy(record),
        score_caption_quality(record),
        score_cross_modal_retrieval(record),
        score_grounding_accuracy(record),
        score_ocr_text_read_accuracy(record),
        score_modality_robustness(record),
        score_cross_modal_hallucination(record),
        score_sub_goal_progress(record),
        score_pass_at_k_agent(record),
        score_consistency(record),
        score_test_retest_stability(record),
        score_termination_correctness(record),
        score_verification_correctness(record),
        score_delegation_accuracy(record),
        score_error_propagation_containment(record),
        score_deadlock_livelock(record),
        score_cumulative_unsafe_action_mas(record),
        score_fairness_gap(record),
        score_protected_attribute_bias(record),
        score_pass_hat_k_agent(record),
        score_bleu(record),
        score_rouge(record),
        score_wer(record),
        score_f1(record),
        score_error_recovery(record),
        score_contamination(record),
        score_unsafe_action(record),
        score_refusal(record),
        score_over_refusal(record),
        score_confirmation_gating(record),
        score_abstention(record),
        score_latency_vs_budget(record),
        score_step_efficiency(record),
        score_cost_per_query(record),
        score_energy_per_query(record),
        score_interruptibility(record),
        score_human_override_success(record),
        score_deferral_on_uncertainty(record),
        score_verbalized_confidence(record),
    ]


def print_metric_map() -> None:
    """Print each card question and its starter metrics. Empty means N/A later."""
    from aixeval.scorecard.profile import DIMENSIONS  # noqa: PLC0415

    print("Questions")
    for question in DIMENSIONS:
        metrics = metrics_for_dimension(question)
        shown = ", ".join(metrics) if metrics else "(none yet)"
        print(f"  {question}: {shown}")


def print_score_metrics_example() -> None:
    """Print one record through score_metrics, then the card."""
    from aixeval.scorecard.profile import build_profile  # noqa: PLC0415

    record = {
        "expected": "yes",
        "output": "yes",
        "grounding_judge": 1.0,
        "had_failure": True,
        "recovered": True,
        "contaminated": False,
        "high_impact_in_scope": False,
        "stated_confidence": 1.0,
    }
    results = score_metrics(record)
    print("score_metrics runs every POSTHOC_METRICS id; opt-in metrics are skipped.")
    print(f"  record:  {record}")
    print(f"  results: {results}")
    print("  card:")
    profile = build_profile(results)
    for question, grade in profile.items():
        shown = "N/A" if grade is None else str(grade)
        print(f"    {question}: {shown}")


if __name__ == "__main__":
    from aixeval.scorecard.sources import print_metric_sources

    print_metric_map()
    print()
    print_metric_sources()
    print()
    print_score_metrics_example()
