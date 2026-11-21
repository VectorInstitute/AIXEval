# Limitations

## AI-judge scores vary between calls

The same prompt, on the same model, can return a different verdict on
a second call. That is a property of the judge, not a bug in the
scorecard.

The toolkit **measures** that variation when you ask it to:

- `test_retest_stability`: variance of one metric on one frozen input
- `cross_run_consistency`: variance across many problems' repeated
  bundles

See [Methodology: consistency tooling](methodology.md#consistency-tooling).
If you only call the judge once, you have not measured the variation.

## Judge models can be wrong

A live-judge metric is an LLM's rating, parsed onto `[0, 1]`. It can
miss a hallucination, invent one, or disagree with a human rater.
Grounding evidence is whatever the ingest path extracted; if that
extract is thin, the judge is grading the wrong text.

Do not treat a judge grade as ground truth. Pair it with declared
gold (`expected`, `should_refuse`, overlays) when you have it.

## Results may not be reproducible once a provider retires a model

Hosted providers change and retire model ids. A bundle that names
`gpt-5.5` or `claude-sonnet-5` cannot be re-judged bit-identically
after that snapshot is gone.

`aixeval replay` still reproduces the **card from the saved record**
(no model call). It cannot recreate a missing vendor model.

Pin an open-weight judge and keep the weights if you need long-term
re-scoring. See [Self-hosting](self-hosting.md).

## Fairness is often N/A

Fairness is **not** a missing dimension. Two metrics exist:

- `fairness_gap`: declared group outcome rates (EEOC four-fifths
  construct). Needs `group_outcomes`.
- `protected_attribute_bias`: judge: did this run resist a declared
  biasing cue? Needs `should_resist_bias` plus a signal and a verdict.

Without that evidence the fairness question is **N/A**. That is
visible on the card, including the F2 stereotype scenarios: they
exercise other metrics; they do not silently invent a fairness score.

There is still no toolkit-computed fairness metric that infers
protected attributes from a single trace. We do not guess group
membership.

## The toolkit only evaluates finished runs

There is no live harness in this repo yet. **`harness/`** and **`serve/`**
are stubs; **Phase 2** will add live harness and hosted serve (see
[Roadmap](about/roadmap.md)). Nothing here confirms a tool call before
it happens, interrupts a running agent, or blocks a deployment.

Upload a dump or trace of a run that already finished.

## Reference images may remain in the OS temp folder

The UI writes an uploaded reference image to a temp file
(`aixeval-image-*`) so the cross-modal judge can read a path. It
deletes that file when you replace or clear the upload in the same
session.

If you close the browser tab mid-session, that cleanup may not run.
The file is a copy of what you uploaded; treat the machine's temp
directory as in-scope for your data-handling rules.

## Indicative API cost for AI-judge metrics

Costs change with provider, model, and how long the trace/evidence is.
These figures are **order-of-magnitude**, not a quote.

| Metric | Live calls | What is sent | Rough cost class (2026 hosted APIs) |
|---|---:|---|---|
| `grounding`, `trajectory_optimality`, `abstention`, `refusal`, `toxicity`, `tool_call_accuracy`, `protected_attribute_bias` | 1 | prompt + output (and evidence or trajectory) | typically well under a few cents per run on mid-size models |
| `pairwise_preference` | 2 | both outputs | about 2× a single text judge |
| `cross_modal_hallucination` | 1 vision call | image + output | higher than text-only |
| Metric suggester | 1 | run facts + your description | similar to one short text judge |
| `--test-retest-metric` | N | same as that metric, N times | N × that metric |

Self-hosted vLLM has no per-token invoice; you pay GPU time instead.
Computed metrics (BLEU, F1, step counts, …) do not call a provider.

Opt-in local models (`unitary/toxic-bert`, DeBERTa for BERTScore,
CLIP) download once and then run on your machine. They are large
(hundreds of MB to multiple GB).
