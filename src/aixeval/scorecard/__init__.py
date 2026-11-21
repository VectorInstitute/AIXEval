"""Eight-question card and posthoc metrics."""

from aixeval.scorecard.abstention import score_abstention
from aixeval.scorecard.bertscore import score_bertscore
from aixeval.scorecard.bleu import score_bleu
from aixeval.scorecard.caption_quality import score_caption_quality
from aixeval.scorecard.clip_score import score_clip_score
from aixeval.scorecard.confirmation_gating import score_confirmation_gating
from aixeval.scorecard.consistency import score_consistency
from aixeval.scorecard.contamination import score_contamination
from aixeval.scorecard.cost_per_query import score_cost_per_query
from aixeval.scorecard.cross_modal_hallucination import score_cross_modal_hallucination
from aixeval.scorecard.cross_modal_retrieval import score_cross_modal_retrieval
from aixeval.scorecard.cumulative_unsafe_action_mas import score_cumulative_unsafe_action_mas
from aixeval.scorecard.deadlock_livelock import score_deadlock_livelock
from aixeval.scorecard.deferral_on_uncertainty import score_deferral_on_uncertainty
from aixeval.scorecard.definitions import (
    DIMENSION_DEFINITIONS,
    HARD_STOP_DEFINITIONS,
    METRIC_DEFINITIONS,
    NA_REASON_DEFINITIONS,
    OPT_IN_METRIC_DEFINITIONS,
    definition_for_any_metric,
    definition_for_metric,
    definition_for_na_reason,
)
from aixeval.scorecard.delegation_accuracy import score_delegation_accuracy
from aixeval.scorecard.energy_per_query import score_energy_per_query
from aixeval.scorecard.error_propagation_containment import score_error_propagation_containment
from aixeval.scorecard.error_recovery import score_error_recovery
from aixeval.scorecard.f1 import score_f1
from aixeval.scorecard.fairness_gap import score_fairness_gap
from aixeval.scorecard.grounding import score_grounding
from aixeval.scorecard.grounding_accuracy import score_grounding_accuracy
from aixeval.scorecard.human_override_success import score_human_override_success
from aixeval.scorecard.interruptibility import score_interruptibility
from aixeval.scorecard.latency_vs_budget import score_latency_vs_budget
from aixeval.scorecard.meteor import score_meteor
from aixeval.scorecard.metrics import (
    ALL_METRIC_IDS,
    METRIC_DIMENSION,
    OPT_IN_METRICS,
    POSTHOC_METRICS,
    metrics_for_dimension,
    score_metrics,
)
from aixeval.scorecard.modality_robustness import score_modality_robustness
from aixeval.scorecard.na_reasons import NA_REASONS, is_na_reason, require_na_reason
from aixeval.scorecard.ocr_text_read_accuracy import score_ocr_text_read_accuracy
from aixeval.scorecard.over_refusal import score_over_refusal
from aixeval.scorecard.pairwise_preference import score_pairwise_preference
from aixeval.scorecard.pass_at_k_agent import score_pass_at_k_agent
from aixeval.scorecard.pass_hat_k_agent import score_pass_hat_k_agent
from aixeval.scorecard.profile import (
    BAND_LABELS,
    BAND_MAX,
    BAND_MIN,
    DIMENSIONS,
    HARD_STOP_RULES,
    HARD_STOPS,
    HIGHER_IS_BETTER_CUTOFFS,
    LOWER_IS_BETTER_CUTOFFS,
    apply_grade,
    build_profile,
    grade_for_raw_score,
    label_for_band,
)
from aixeval.scorecard.protected_attribute_bias import score_protected_attribute_bias
from aixeval.scorecard.refusal import score_refusal
from aixeval.scorecard.rouge import score_rouge
from aixeval.scorecard.sources import DIMENSION_FRAME, METRIC_SOURCES
from aixeval.scorecard.step_efficiency import score_step_efficiency
from aixeval.scorecard.sub_goal_progress import score_sub_goal_progress
from aixeval.scorecard.task_success import score_task_success
from aixeval.scorecard.termination_correctness import score_termination_correctness
from aixeval.scorecard.test_retest_stability import score_test_retest_stability
from aixeval.scorecard.tool_call_accuracy import score_tool_call_accuracy
from aixeval.scorecard.toxicity import score_toxicity
from aixeval.scorecard.trajectory_optimality import score_trajectory_optimality
from aixeval.scorecard.unsafe_action import score_unsafe_action
from aixeval.scorecard.verbalized_confidence import score_verbalized_confidence
from aixeval.scorecard.verification_correctness import score_verification_correctness
from aixeval.scorecard.vqa_accuracy import score_vqa_accuracy
from aixeval.scorecard.wer import score_wer


__all__ = [
    "ALL_METRIC_IDS",
    "BAND_LABELS",
    "BAND_MAX",
    "BAND_MIN",
    "DIMENSIONS",
    "DIMENSION_DEFINITIONS",
    "DIMENSION_FRAME",
    "HARD_STOP_DEFINITIONS",
    "HARD_STOP_RULES",
    "HARD_STOPS",
    "HIGHER_IS_BETTER_CUTOFFS",
    "LOWER_IS_BETTER_CUTOFFS",
    "METRIC_DEFINITIONS",
    "METRIC_DIMENSION",
    "METRIC_SOURCES",
    "NA_REASON_DEFINITIONS",
    "NA_REASONS",
    "OPT_IN_METRICS",
    "OPT_IN_METRIC_DEFINITIONS",
    "POSTHOC_METRICS",
    "apply_grade",
    "build_profile",
    "definition_for_any_metric",
    "definition_for_metric",
    "definition_for_na_reason",
    "grade_for_raw_score",
    "is_na_reason",
    "label_for_band",
    "metrics_for_dimension",
    "require_na_reason",
    "score_abstention",
    "score_bertscore",
    "score_bleu",
    "score_caption_quality",
    "score_clip_score",
    "score_confirmation_gating",
    "score_contamination",
    "score_consistency",
    "score_delegation_accuracy",
    "score_error_propagation_containment",
    "score_deadlock_livelock",
    "score_cumulative_unsafe_action_mas",
    "score_cost_per_query",
    "score_cross_modal_hallucination",
    "score_cross_modal_retrieval",
    "score_deferral_on_uncertainty",
    "score_energy_per_query",
    "score_error_recovery",
    "score_f1",
    "score_fairness_gap",
    "score_grounding",
    "score_grounding_accuracy",
    "score_human_override_success",
    "score_interruptibility",
    "score_latency_vs_budget",
    "score_meteor",
    "score_metrics",
    "score_modality_robustness",
    "score_ocr_text_read_accuracy",
    "score_over_refusal",
    "score_pairwise_preference",
    "score_pass_at_k_agent",
    "score_pass_hat_k_agent",
    "score_protected_attribute_bias",
    "score_refusal",
    "score_rouge",
    "score_step_efficiency",
    "score_sub_goal_progress",
    "score_task_success",
    "score_termination_correctness",
    "score_test_retest_stability",
    "score_tool_call_accuracy",
    "score_toxicity",
    "score_trajectory_optimality",
    "score_unsafe_action",
    "score_verbalized_confidence",
    "score_verification_correctness",
    "score_vqa_accuracy",
    "score_wer",
]
