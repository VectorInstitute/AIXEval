---
hide:
  - title
---

<div class="aix-page-hero aix-page-hero--purple" markdown="0">
  <p class="aix-page-hero__label">Metric library</p>
  <h1>Metric catalog</h1>
  <p class="aix-page-hero__lead">47 metric ids, mechanisms, and required record fields.</p>
</div>

47 metric ids. Missing evidence is **N/A**, never a silent 0.

Mechanisms:

- **declared**: reads an asserted field (ingest never guesses it)
- **computed**: deterministic formula or library comparison
- **judge**: live LLM call
- **overlay**: declared gold cross-checked against the trace (multi-agent)

Rationale, citations, and edge cases: `src/aixeval/scorecard/<metric_id>.py`.

## Card metrics (`score_metrics()`)

| Dimension | Metric | Mechanism | Needs |
|---|---|---|---|
| capability | `task_success` | computed | `expected`, `output` |
| capability | `grounding` | judge | trajectory, `output` |
| capability | `bleu` / `rouge` / `wer` / `f1` | computed | `expected`, `output` |
| capability | `trajectory_optimality` | judge | trajectory, `output` |
| capability | `tool_call_accuracy` | judge **or** computed | trajectory + `gold_tool_plan`, **or** `gold_tool_calls` |
| capability | `pairwise_preference` | judge (2 calls) | `output`, `baseline_output` |
| capability | `vqa_accuracy` | computed | `output`, `gold_answers` |
| capability | `caption_quality` | computed | `output`, `reference_captions` |
| capability | `cross_modal_retrieval` | declared | `retrieval_rank`, `retrieval_k` |
| capability | `grounding_accuracy` | computed | `predicted_bbox`, `gold_bbox` |
| capability | `ocr_text_read_accuracy` | computed | `expected`, `output` |
| capability | `sub_goal_progress` | declared | `subgoals_completed`, `subgoals_total` |
| capability | `pass_at_k_agent` | declared | `attempt_budget`, `solved_within_budget` |
| capability | `pass_hat_k_agent` | computed | `pass_hat_k_attempts`, `pass_hat_k_successes` |
| capability | `delegation_accuracy` | overlay | `gold_roles` + trace |
| robustness | `error_recovery` | declared | `had_failure`, `recovered` |
| robustness | `modality_robustness` | declared | `clean_score`, `perturbed_score` |
| robustness | `consistency` | declared | `run_a_score`, `run_b_score` |
| robustness | `test_retest_stability` | declared | `rerun_scores` |
| robustness | `termination_correctness` | declared | `terminated_correctly` |
| robustness | `verification_correctness` | declared | `verified_correctly` |
| robustness | `error_propagation_containment` | overlay | `faults` + trace |
| robustness | `deadlock_livelock` | overlay | `run_stalled` |
| safety | `cumulative_unsafe_action_mas` | overlay | `mas_unsafe_steps`/`mas_total_steps` |
| safety | `unsafe_action` | declared | `high_impact_in_scope`, `unsafe_taken` |
| safety | `refusal` / `over_refusal` | judge (shared verdict) | `output`, gold `should_refuse` |
| safety | `cross_modal_hallucination` | judge | `output`, `image_path` or `image_description` |
| fairness | `fairness_gap` | declared | `group_outcomes` |
| fairness | `protected_attribute_bias` | judge | `should_resist_bias`, `protected_attribute_signal` + `output` |
| transparency | `verbalized_confidence` | declared | `stated_confidence` |
| governance | `contamination` | declared **or** audited | `contaminated`, or `expected` + trace input text |
| oversight | `confirmation_gating` | declared | `high_impact_in_scope`, `confirmed` |
| oversight | `abstention` | judge | `output`, gold `should_abstain` |
| oversight | `interruptibility` | declared | `stop_signal_sent`, `halted_cleanly` |
| oversight | `human_override_success` | declared | `override_requested`, `override_applied` |
| oversight | `deferral_on_uncertainty` | declared | `low_confidence_flagged`, `deferred_to_human` |
| efficiency | `latency_vs_budget` | declared | `latency_sec`, `latency_budget_sec` |
| efficiency | `step_efficiency` | declared | `steps`, `step_budget` |
| efficiency | `cost_per_query` | declared | `cost_usd`, `cost_budget_usd` |
| efficiency | `energy_per_query` | declared | `energy_kwh`, `energy_budget_kwh` |

## Opt-in (call directly: not in `score_metrics()`)

Heavy one-time downloads where noted.

| Metric | Mechanism | Needs |
|---|---|---|
| `toxicity` | judge **or** classifier (`unitary/toxic-bert`) | `output` |
| `bertscore` | computed (DeBERTa contextual F1) | `expected`, `output` |
| `meteor` | computed (NLTK; NLTK corpora download) | `expected`, `output` |
| `clip_score` | computed (CLIP embedding similarity) | `output`, `image_path` |

## Ingest shapes

- **Dumps**: plain JSON records (`ingest/dumps.py`)
- **Traces**: Langfuse, OpenTelemetry, Patronus-shaped, tau-bench via
  `ingest/pipeline.ingest_any`
- **Inspect AI**: `.eval` archives via `ingest_inspect_eval`

Declared fields and multi-agent overlays merge on top; see
[`user_guide.md`](user_guide.md) and the `ingest/` package.

Literature vs policy tags for each metric: run
`uv run python -m aixeval.scorecard.sources`.

To contribute a new id: [Add a metric](extending/add-a-metric.md).
