"""LLM judge runners. Some read the original dump, some only the sealed record."""

from aixeval.ingest.trace_taubench import extract_tool_calls_taubench
from aixeval.judge.abstention import judge_abstention
from aixeval.judge.client import ClaudeClient
from aixeval.judge.config import build_client, build_client_from_env
from aixeval.judge.cross_modal_hallucination import judge_cross_modal_hallucination
from aixeval.judge.evidence import (
    extract_evidence_langfuse,
    extract_tool_calls_langfuse,
    extract_trajectory_langfuse,
    join_evidence_fields,
)
from aixeval.judge.gemini_client import GeminiClient
from aixeval.judge.grounding import judge_grounding
from aixeval.judge.live import (
    LIVE_JUDGE_METRICS,
    extract_live_grounding_evidence,
    live_judge_is_possible,
    run_live_judges,
)
from aixeval.judge.metric_suggester import suggest_relevant_metrics
from aixeval.judge.openai_client import OpenAIClient
from aixeval.judge.pairwise_preference import judge_pairwise_preference
from aixeval.judge.protected_attribute_bias import (
    judge_detect_protected_attribute_signal,
    judge_protected_attribute_bias,
)
from aixeval.judge.provenance import merge_judge_provenance, models_from_provenance
from aixeval.judge.provider import Provider
from aixeval.judge.refusal import judge_refusal
from aixeval.judge.runner import (
    CapturingProvider,
    collect_rerun_scores,
    run_abstention_judge,
    run_cross_modal_hallucination_judge,
    run_grounding_judge,
    run_pairwise_preference_judge,
    run_protected_attribute_bias_detector,
    run_protected_attribute_bias_judge,
    run_refusal_judge,
    run_tool_call_accuracy_check,
    run_tool_call_accuracy_judge,
    run_toxicity_judge,
    run_trajectory_optimality_judge,
)
from aixeval.judge.tool_call_accuracy import judge_tool_call_accuracy
from aixeval.judge.toxicity import judge_toxicity
from aixeval.judge.trajectory_optimality import judge_trajectory_optimality
from aixeval.judge.vllm_client import VLLMClient


__all__ = [
    "CapturingProvider",
    "ClaudeClient",
    "GeminiClient",
    "OpenAIClient",
    "Provider",
    "VLLMClient",
    "build_client",
    "build_client_from_env",
    "collect_rerun_scores",
    "extract_evidence_langfuse",
    "extract_tool_calls_langfuse",
    "extract_tool_calls_taubench",
    "extract_trajectory_langfuse",
    "join_evidence_fields",
    "LIVE_JUDGE_METRICS",
    "extract_live_grounding_evidence",
    "live_judge_is_possible",
    "merge_judge_provenance",
    "models_from_provenance",
    "judge_abstention",
    "judge_cross_modal_hallucination",
    "judge_detect_protected_attribute_signal",
    "judge_grounding",
    "judge_pairwise_preference",
    "judge_protected_attribute_bias",
    "judge_refusal",
    "judge_tool_call_accuracy",
    "judge_toxicity",
    "judge_trajectory_optimality",
    "run_abstention_judge",
    "run_cross_modal_hallucination_judge",
    "run_grounding_judge",
    "run_live_judges",
    "run_pairwise_preference_judge",
    "run_protected_attribute_bias_detector",
    "run_protected_attribute_bias_judge",
    "run_refusal_judge",
    "run_tool_call_accuracy_check",
    "run_tool_call_accuracy_judge",
    "run_toxicity_judge",
    "run_trajectory_optimality_judge",
    "suggest_relevant_metrics",
]
