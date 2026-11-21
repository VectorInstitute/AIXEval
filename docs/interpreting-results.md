# Interpreting results

## What a score means

A grade is a **post-hoc reading of one saved run** against the metrics
you selected and the policy in force (cutoffs and hard stops).

- `4/4 (Excellent)` on a question means every *scored* metric under
  that question cleared the Excellent band. Metrics that were N/A did
  not vote.
- `N/A` means this toolkit did not have the evidence that metric
  needs (no gold answer, no overlay, no judge verdict, cancelled
  record, …). It is not a pass and not a fail.
- The eight-question card is a **minimum**: one weak metric pulls the
  whole question down.

Read the per-metric table, not only the card. The card is a summary.

## What a score does not mean

A high score is **not** evidence that a system is safe to deploy.

It is also not:

- a certification, including under the EU AI Act
- a security assessment or penetration test
- a substitute for domain-specific safety testing (clinical, hiring,
  legal, or otherwise)
- a live safety control: nothing here blocks an agent in production
- a basis for decisions about people

Hard stops and N/A rows are part of the result. Ignoring them and
quoting only the pretty grades is a misuse of the card.

## How to read hard stops

Hard stops are applied **after** metrics are scored. They change the
card; they never rewrite `raw_score`.

| What you see | What it means |
|---|---|
| `hard stops: (none)` | No shipped rule fired. |
| `hard stops: unsafe_action_cap` | The unsafe-action metric cleared its cutoff; filled questions were capped. |
| `uncapped (before hard-stop caps): efficiency 4/4` | Efficiency was 4 before the cap; the 0 you see is the cap, not the metric. |
| Capability `N/A` after `contamination_void` | Capability was withdrawn, not “failed at 0”. |

The CLI prints the rule, the cutoff, and the record fields that fired
it. The UI shows capped and uncapped values side by side and, when you
run incident analysis, a plain-English pointer at the trace step.

Treat the hard-stop line as a **separate finding** from the uncapped
grades. Do not average them away.

Change the policy without rescoring: see
[Methodology: hard-stop cutoffs](methodology.md#hard-stop-cutoffs-declared-policy).

## Do not compare grades across judge models

Live-judge metrics (`grounding`, `trajectory_optimality`, `refusal`,
…) are the opinion of **that** model, on **that** prompt version.

- Two runs judged by different models are not the same measurement.
- The same model on a later provider snapshot may not match an older
  bundle.
- Replay reuses saved verdicts; turning replay off and scoring live
  with a new sidebar judge is a new measurement.

The card and the run bundle name the judge when one was used. Compare
only names a divergence; it does not tell you which judge is “right”.

If you need a number you can re-grade later without a vendor, prefer
computed and declared metrics, or a [self-hosted judge](self-hosting.md)
whose weights you pin.
