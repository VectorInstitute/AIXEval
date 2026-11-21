---
hide:
  - title
---

<div class="aix-page-hero aix-page-hero--magenta" markdown="0">
  <p class="aix-page-hero__label">Concepts</p>
  <h1>How the scorecard works</h1>
  <p class="aix-page-hero__lead">Dimensions, bands, hard stops, and how to read grades.</p>
</div>

These pages describe **what the scorecard measures** and **how to read it**.
They are aimed at evaluators and tech leads, not only Python contributors.

## Flow (post-hoc)

1. **Ingest**: normalize a simple JSON dump or a trace (Langfuse, OTel,
   Patronus, Inspect AI, and others) into one sealed record per run.
2. **Score**: run metrics from the catalog; missing evidence is **N/A**, not a silent 0.
3. **Card**: roll metric grades into eight questions (0–4 each), then apply
   declared **hard stops** (contamination, unsafe action, confirmation gating).

The toolkit never rewrites a raw metric score when a hard stop fires; it
only caps or voids card questions per policy.

## Read next

| Topic | Page |
|---|---|
| Dimensions, bands, literature vs policy | [Methodology](../methodology.md) |
| What a high grade does *not* mean | [Interpreting results](../interpreting-results.md) |
| Judge variance, fairness N/A, cost | [Limitations](../limitations.md) |
| All metric ids and required fields | [Metric catalog](../metric-catalog.md) |
