---
hide:
  - title
---

<div class="aix-page-hero aix-page-hero--magenta" markdown="0">
  <p class="aix-page-hero__label">About</p>
  <h1>AIXeval</h1>
  <p class="aix-page-hero__lead">Vector Institute open toolkit for post-hoc agent evaluation.</p>
</div>

**AIXeval** is an open post-hoc evaluation toolkit from the
[Vector Institute](https://vectorinstitute.ai). It turns a **saved**
agent run (a JSON dump or an exported trace) into a structured
scorecard: dozens of named metrics rolled into eight trustworthiness
questions, each graded 0–4.

Development is informed by agent-evaluation practice (HELM-style
multi-facet reporting), governance framing (NIST AI RMF), and the
eight-dimension synthesis in
[Raza et al. (2026)](https://arxiv.org/abs/2609.19524). Details:
[Methodology](../methodology.md).

## What it is

- A **library** (`aixeval`), **CLI**, and optional **Streamlit UI**
- **Ingest** for common dump and trace shapes (Langfuse, OpenTelemetry,
  Patronus-shaped JSON, Inspect AI `.eval`, and others)
- **Judges** via Anthropic, OpenAI, Google Gemini, or self-hosted vLLM
  for metrics that need a live model call

## What it is not

AIXeval does **not**:

- Execute or orchestrate your agent during deployment
- Act as a live safety filter or policy engine
- Certify compliance (including EU AI Act), replace penetration testing,
  or substitute domain-specific safety work
- Support decisions about people (hiring, medical care, legal outcomes)

A strong card is **not** evidence that a system is safe to deploy. Read
[Interpreting results](../interpreting-results.md) and
[Limitations](../limitations.md).

## Project context

This repository is the public-facing **`aixeval`** package. It supports
research and engineering evaluation workflows; it is maintained by Vector
AI Engineering with community contributions via GitHub.

| Topic | Page |
|---|---|
| Phase 1 vs Phase 2 (harness, serve) | [Roadmap](roadmap.md) |
| Funding and sponsors | [Acknowledgments](acknowledgments.md) |
| Licence, trademarks, how to cite | [Licence and citation](license-and-citation.md) |
| Contact and pull requests | [Contributing](../extending/contributing.md) |
| Security reports | [`SECURITY.md`](https://github.com/VectorInstitute/aixeval/blob/main/SECURITY.md) on GitHub |

Support triage: `shaina.raza@vectorinstitute.ai`.