"""Sealed record: closed field names and types. Missing keys are allowed."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any, TypedDict


class SealedRecord(TypedDict, total=False):
    """Fields ``score_metrics`` already reads. Omit a key if unknown."""

    expected: str
    output: str
    vetoed: bool
    grounding_judge: float
    had_failure: bool
    recovered: bool
    contaminated: bool
    high_impact_in_scope: bool
    unsafe_taken: bool
    confirmed: bool
    latency_sec: float
    latency_budget_sec: float
    steps: int
    step_budget: int
    stated_confidence: float
    cancelled: bool
    evidence_status: str
    should_abstain: bool
    abstention_judge: float
    should_refuse: bool
    refusal_judge: float
    trajectory_optimality_judge: float
    gold_tool_calls: list[dict[str, Any]]
    gold_tool_plan: str
    tool_call_accuracy_judge: float
    cost_usd: float
    cost_budget_usd: float
    toxicity_judge: float
    energy_kwh: float
    energy_budget_kwh: float
    stop_signal_sent: bool
    halted_cleanly: bool
    override_requested: bool
    override_applied: bool
    low_confidence_flagged: bool
    deferred_to_human: bool
    baseline_output: str
    pairwise_preference_judge: float
    retrieval_rank: int
    retrieval_k: int
    predicted_bbox: list[float]
    gold_bbox: list[float]
    clean_score: float
    perturbed_score: float
    gold_answers: list[str]
    reference_captions: list[str]
    image_description: str
    cross_modal_hallucination_judge: float
    image_path: str
    subgoals_completed: int
    subgoals_total: int
    attempt_budget: int
    solved_within_budget: bool
    run_a_score: float
    run_b_score: float
    terminated_correctly: bool
    verified_correctly: bool
    faults_total: int
    faults_contained: int
    subtasks_total: int
    subtasks_correct: int
    run_stalled: bool
    mas_unsafe_steps: int
    mas_total_steps: int
    rerun_scores: list[float]
    group_outcomes: dict[str, float]
    should_resist_bias: bool
    protected_attribute_signal: str
    protected_attribute_bias_judge: float
    pass_hat_k_attempts: int
    pass_hat_k_successes: int
    judge_provenance: dict[str, dict[str, str]]


RECORD_FIELDS: tuple[str, ...] = tuple(SealedRecord.__optional_keys__)


def validate_record(record: dict[str, Any]) -> dict[str, Any]:
    """Check names and types. Unknown keys and bad types raise ``ValueError``.

    Absence is legal. This does not check that a translator mapped the
    right source field.
    """
    unknown = sorted(set(record) - set(RECORD_FIELDS))
    if unknown:
        shown = ", ".join(unknown)
        msg = f"Unknown sealed-record field(s): {shown}"
        raise ValueError(msg)

    checked: dict[str, Any] = {}
    for key, value in record.items():
        checked[key] = _check_field(key, value)
    return checked


def _check_field(key: str, value: object) -> object:
    validator = _VALIDATORS.get(key)
    if validator is None:
        msg = f"No type rule for field {key!r}"
        raise ValueError(msg)
    return validator(key, value)


def _as_str(key: str, value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, str):
        msg = f"{key} must be text"
        raise ValueError(msg)
    return value


def _as_bool(key: str, value: object) -> bool:
    if not isinstance(value, bool):
        msg = f"{key} must be a bool"
        raise ValueError(msg)
    return value


def _as_unit(key: str, value: object) -> float:
    number = _as_finite_float(key, value)
    if number < 0.0 or number > 1.0:
        msg = f"{key} must be on [0, 1]"
        raise ValueError(msg)
    return number


def _as_nonneg_float(key: str, value: object) -> float:
    number = _as_finite_float(key, value)
    if number < 0.0:
        msg = f"{key} must be >= 0"
        raise ValueError(msg)
    return number


def _as_finite_float(key: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{key} must be a finite number"
        raise ValueError(msg)
    number = float(value)
    if not isfinite(number):
        msg = f"{key} must be a finite number"
        raise ValueError(msg)
    return number


def _as_nonneg_int(key: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{key} must be an int >= 0"
        raise ValueError(msg)
    if value < 0:
        msg = f"{key} must be an int >= 0"
        raise ValueError(msg)
    return value


def _as_tool_call_list(key: str, value: object) -> list[dict[str, Any]]:
    """Check a list of ``{"tool": <non-empty str>, "args": <object>}`` dicts.

    The first non-scalar field type in this schema; everything else
    is str/bool/float/int. ``args`` defaults to ``{}`` if omitted, so
    downstream comparison never has to special-case a missing one.
    """
    if not isinstance(value, list):
        msg = f"{key} must be a list"
        raise ValueError(msg)
    checked: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            msg = f"{key} items must be objects with a 'tool' name"
            raise ValueError(msg)
        tool = item.get("tool")
        if not isinstance(tool, str) or not tool:
            msg = f"{key} items need a non-empty string 'tool'"
            raise ValueError(msg)
        args = item.get("args", {})
        if not isinstance(args, dict):
            msg = f"{key} items' 'args' must be an object"
            raise ValueError(msg)
        checked.append({"tool": tool, "args": args})
    return checked


def _as_bbox(key: str, value: object) -> list[float]:
    """Check a ``[x1, y1, x2, y2]`` list of exactly 4 finite floats.

    No ordering constraint enforced here (``x2 > x1`` etc.): a
    degenerate box collapses to zero area/IoU downstream, which is
    enough signal on its own, without a second place this can fail.
    """
    if not isinstance(value, list) or len(value) != 4:
        msg = f"{key} must be a list of exactly 4 numbers [x1, y1, x2, y2]"
        raise ValueError(msg)
    return [_as_finite_float(key, coordinate) for coordinate in value]


def _as_str_list(key: str, value: object) -> list[str]:
    """Check a list of non-empty strings."""
    if not isinstance(value, list):
        msg = f"{key} must be a list"
        raise ValueError(msg)
    checked: list[str] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, str) or not item:
            msg = f"{key} items must be non-empty strings"
            raise ValueError(msg)
        checked.append(item)
    return checked


def _as_unit_list(key: str, value: object) -> list[float]:
    """Check a list of at least 2 finite numbers on [0, 1]."""
    min_items = 2
    if not isinstance(value, list) or len(value) < min_items:
        msg = f"{key} must be a list of at least {min_items} numbers on [0, 1]"
        raise ValueError(msg)
    return [_as_unit(key, item) for item in value]


def _as_group_outcomes(key: str, value: object) -> dict[str, float]:
    """Check a mapping of at least 2 group labels to finite numbers on [0, 1]."""
    min_items = 2
    if not isinstance(value, dict) or len(value) < min_items:
        msg = f"{key} must be an object mapping at least {min_items} group labels to numbers on [0, 1]"
        raise ValueError(msg)
    checked: dict[str, float] = {}
    for label, item in value.items():
        if not isinstance(label, str) or not label:
            msg = f"{key} keys must be non-empty strings"
            raise ValueError(msg)
        checked[label] = _as_unit(key, item)
    return checked


def _as_judge_provenance(key: str, value: object) -> dict[str, dict[str, str]]:
    """Check a mapping of metric ids to ``{model, prompt_version}``."""
    if not isinstance(value, dict) or not value:
        msg = f"{key} must be a non-empty object mapping metric ids to provenance"
        raise ValueError(msg)
    checked: dict[str, dict[str, str]] = {}
    for metric_id, entry in value.items():
        if not isinstance(metric_id, str) or not metric_id:
            msg = f"{key} keys must be non-empty strings"
            raise ValueError(msg)
        if not isinstance(entry, dict):
            msg = f"{key}[{metric_id!r}] must be an object"
            raise ValueError(msg)
        model = entry.get("model")
        prompt_version = entry.get("prompt_version")
        if not isinstance(model, str) or not model.strip():
            msg = f"{key}[{metric_id!r}].model must be non-empty text"
            raise ValueError(msg)
        if not isinstance(prompt_version, str) or not prompt_version.strip():
            msg = f"{key}[{metric_id!r}].prompt_version must be non-empty text"
            raise ValueError(msg)
        checked[metric_id] = {"model": model, "prompt_version": prompt_version}
    return checked


_TEXT = frozenset(
    {
        "expected",
        "output",
        "evidence_status",
        "gold_tool_plan",
        "baseline_output",
        "image_description",
        "image_path",
        "protected_attribute_signal",
    }
)
_BOOL = frozenset(
    {
        "vetoed",
        "had_failure",
        "recovered",
        "contaminated",
        "high_impact_in_scope",
        "unsafe_taken",
        "confirmed",
        "cancelled",
        "should_abstain",
        "should_refuse",
        "stop_signal_sent",
        "halted_cleanly",
        "override_requested",
        "override_applied",
        "low_confidence_flagged",
        "deferred_to_human",
        "solved_within_budget",
        "terminated_correctly",
        "verified_correctly",
        "run_stalled",
        "should_resist_bias",
    }
)
_UNIT = frozenset(
    {
        "grounding_judge",
        "stated_confidence",
        "abstention_judge",
        "refusal_judge",
        "trajectory_optimality_judge",
        "tool_call_accuracy_judge",
        "toxicity_judge",
        "pairwise_preference_judge",
        "clean_score",
        "perturbed_score",
        "cross_modal_hallucination_judge",
        "run_a_score",
        "run_b_score",
        "protected_attribute_bias_judge",
    }
)
_NONNEG_FLOAT = frozenset(
    {"latency_sec", "latency_budget_sec", "cost_usd", "cost_budget_usd", "energy_kwh", "energy_budget_kwh"}
)
_NONNEG_INT = frozenset(
    {
        "steps",
        "step_budget",
        "retrieval_rank",
        "retrieval_k",
        "subgoals_completed",
        "subgoals_total",
        "attempt_budget",
        "faults_total",
        "faults_contained",
        "subtasks_total",
        "subtasks_correct",
        "mas_unsafe_steps",
        "mas_total_steps",
        "pass_hat_k_attempts",
        "pass_hat_k_successes",
    }
)
_TOOL_CALL_LIST = frozenset({"gold_tool_calls"})
_BBOX = frozenset({"predicted_bbox", "gold_bbox"})
_STR_LIST = frozenset({"gold_answers", "reference_captions"})
_UNIT_LIST = frozenset({"rerun_scores"})
_GROUP_OUTCOMES = frozenset({"group_outcomes"})
_JUDGE_PROVENANCE = frozenset({"judge_provenance"})

# One validator per field, built from the type buckets above: a dict
# lookup, not a growing if-chain, so adding a new bucket never risks
# tripping a too-many-branches lint cap again.
_VALIDATORS: dict[str, Callable[[str, object], object]] = {
    **dict.fromkeys(_TEXT, _as_str),
    **dict.fromkeys(_BOOL, _as_bool),
    **dict.fromkeys(_UNIT, _as_unit),
    **dict.fromkeys(_NONNEG_FLOAT, _as_nonneg_float),
    **dict.fromkeys(_NONNEG_INT, _as_nonneg_int),
    **dict.fromkeys(_TOOL_CALL_LIST, _as_tool_call_list),
    **dict.fromkeys(_BBOX, _as_bbox),
    **dict.fromkeys(_STR_LIST, _as_str_list),
    **dict.fromkeys(_UNIT_LIST, _as_unit_list),
    **dict.fromkeys(_GROUP_OUTCOMES, _as_group_outcomes),
    **dict.fromkeys(_JUDGE_PROVENANCE, _as_judge_provenance),
}
