---
hide:
  - title
---

<div class="aix-page-hero aix-page-hero--teal" markdown="0">
  <p class="aix-page-hero__label">Get started</p>
  <h1>Install and run</h1>
  <p class="aix-page-hero__lead">Score a saved run from the CLI, Python, or Streamlit.</p>
</div>

AIXeval scores a **saved** agent run after the fact. It does not execute
your agent or act as a live safety gate.

## What you need

- Python **≥ 3.12**
- [uv](https://docs.astral.sh/uv/) (recommended) or another way to install the package from source
- Optional: API keys or a self-hosted judge for **live-judge** metrics

## Paths in

| Goal | Page |
|---|---|
| Install, Streamlit UI, CLI, tests | [Setup (user guide)](../user_guide.md) |
| Eight dimensions and grading policy | [Methodology](../methodology.md) |
| Pick metrics and required fields | [Metric catalog](../metric-catalog.md) |

## Minimal CLI flow

```bash
uv sync --dev
uv run aixeval score fixtures/simple_match.json --out run.json
uv run aixeval report run.json
```

Details, optional dependency groups, and integration tests: [user guide](../user_guide.md).

## Before you upload traces

Read [Privacy and data](../privacy-and-data.md). Do not send personal,
confidential, or regulated content to a hosted judge without authorization.
