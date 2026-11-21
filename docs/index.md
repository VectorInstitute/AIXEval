---
hide:
  - title
  - navigation
  - toc
---

<div class="aix-hero" markdown="0">
  <div class="aix-hero__mesh" aria-hidden="true"></div>
  <div class="aix-hero__inner">
    <div class="aix-hero__copy">
      <p class="aix-hero__eyebrow">Post-hoc scorecard toolkit</p>
      <h1 class="aix-hero__title">
        Evaluation for <em class="aix-hero__accent">saved agent runs</em>
      </h1>
      <p class="aix-hero__lead">
        Ingest a dump or trace, score 47 metrics, and read an eight-question
        trustworthiness card. Library, CLI, and optional Streamlit UI. Does not
        execute your agent or act as a live safety control.
      </p>
      <div class="aix-hero__cta">
        <a class="aix-btn aix-btn--primary" href="getting-started/">Get started</a>
        <a class="aix-btn aix-btn--ghost" href="metric-catalog/">Metric catalog</a>
        <a class="aix-btn aix-btn--ghost" href="https://github.com/VectorInstitute/AIXEval">GitHub</a>
        <a class="aix-btn aix-btn--ghost" href="https://arxiv.org/abs/2609.19524" target="_blank" rel="noopener noreferrer">To read</a>
      </div>
      <div class="aix-hero__tags">
        <span class="aix-tag">Python &ge; 3.12</span>
        <span class="aix-tag">Apache 2.0</span>
        <span class="aix-tag">47 metrics</span>
      </div>
    </div>
    <div class="aix-hero__visual">
      <img src="assets/aixeval-hero-flow.svg" alt="Ingest, trace, metrics, and judge roll into an eight-question card" width="320" height="320" loading="eager"/>
    </div>
  </div>
</div>

<div class="aix-callout-warning" markdown="0">
  <strong>Before you upload traces.</strong>
  Do not send personal, confidential, or regulated data to a hosted judge without authorization.
  <a href="privacy-and-data/">Privacy and data</a>
</div>

## Explore

<div class="grid cards aix-cards" markdown>

-   :material-rocket-launch-outline:{ .lg .middle } __Get started__

    ---

    Install, UI, CLI, and a minimal score flow

    [:octicons-arrow-right-24: Overview](getting-started/index.md)

-   :material-brain:{ .lg .middle } __Concepts__

    ---

    Eight dimensions, bands, hard stops, how to read grades

    [:octicons-arrow-right-24: Overview](concepts/index.md)

-   :material-table-large:{ .lg .middle } __Metric library__

    ---

    All metric ids, mechanisms, and required record fields

    [:octicons-arrow-right-24: Catalog](metric-catalog.md)

-   :material-cog-outline:{ .lg .middle } __How-to__

    ---

    Judges, self-hosting vLLM, privacy

    [:octicons-arrow-right-24: Guides](how-to/index.md)

-   :material-source-branch:{ .lg .middle } __Extend__

    ---

    Add a metric, contribute, run tests

    [:octicons-arrow-right-24: Overview](extending/index.md)

-   :material-information-outline:{ .lg .middle } __About__

    ---

    Acknowledgments, licence, citation, project context

    [:octicons-arrow-right-24: About AIXeval](about/index.md)

</div>

## How scoring works

<div class="aix-steps" markdown="0">
  <article class="aix-step">
    <span class="aix-step__num">1</span>
    <span class="aix-step__title">Ingest</span>
    <span class="aix-step__body">Normalize a dump or trace into one sealed record.</span>
  </article>
  <article class="aix-step">
    <span class="aix-step__num">2</span>
    <span class="aix-step__title">Score</span>
    <span class="aix-step__body">Run metrics. Missing evidence is N/A, never a silent 0.</span>
  </article>
  <article class="aix-step">
    <span class="aix-step__num">3</span>
    <span class="aix-step__title">Card</span>
    <span class="aix-step__body">Aggregate into eight questions; hard stops may cap questions without changing raw scores.</span>
  </article>
</div>

[Methodology](methodology.md) for dimensions, bands, and policy cutoffs.

## Implementation status

<div class="aix-status-grid" markdown="0">
  <article class="aix-status-card aix-status-card--done"><h3>scorecard</h3><p>47 metrics, grading, eight-question card</p></article>
  <article class="aix-status-card aix-status-card--done"><h3>ingest</h3><p>Dumps, Langfuse, OTel, Patronus, Inspect AI, overlay</p></article>
  <article class="aix-status-card aix-status-card--done"><h3>judge</h3><p>Claude, OpenAI, Gemini, self-hosted vLLM</p></article>
  <article class="aix-status-card aix-status-card--done"><h3>ui</h3><p>Streamlit, post-hoc; sidebar picks live judge</p></article>
  <article class="aix-status-card aix-status-card--stub"><h3>harness</h3><p>Phase 2 (planned): live eval harness; stub today</p></article>
  <article class="aix-status-card aix-status-card--stub"><h3>serve</h3><p>Phase 2 (planned): hosted scorecard service; stub today</p></article>
</div>

Phase 2 scope: [Roadmap](about/roadmap.md). [API reference](api.md) for the Python package.

## Acknowledgments

Research supported by the **Province of Ontario**, **CIFAR**, Vector industry
sponsors, and the EU **Horizon AIXPERT** project
([101214389](https://cordis.europa.eu/project/id/101214389)).
[Full acknowledgments](about/acknowledgments.md) ·
[Licence and citation](about/license-and-citation.md)

**Correspondence:** [Email](mailto:shaina.raza@vectorinstitute.ai)

