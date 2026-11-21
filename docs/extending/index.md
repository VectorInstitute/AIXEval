---
hide:
  - title
---

<div class="aix-page-hero aix-page-hero--blue" markdown="0">
  <p class="aix-page-hero__label">Extend</p>
  <h1>Contribute and customize</h1>
  <p class="aix-page-hero__lead">Add metrics upstream or run private scorers alongside the card.</p>
</div>

The core package ships **47 named metrics** and an eight-question card.
You can extend it in two ways:

## Contribute a first-party metric

Add a scorer under `src/aixeval/scorecard/`, register it in the central
catalog tables, and open a pull request. The maintainer checklist lives in
[Add a metric](add-a-metric.md).

There is **no runtime plugin hook** today: upstream metrics are wired
explicitly in `score_metrics()`. A registry or entry-point design may come
later if third-party catalogs become common.

## Use a private metric (no upstream merge)

Implement a scorer that returns the same result shape as built-in metrics
(`metric_id`, `status`, `raw_score` or `na_reason`, `grade` when scored).
Combine its rows with `score_metrics()` output before `build_profile()`, or
keep a separate report outside the eight-question card. See
[Add a metric: private metrics](add-a-metric.md#private-metrics-without-upstream-merge).

## Contributing code and docs

[Contributing](contributing.md): style, tests, and review expectations.
