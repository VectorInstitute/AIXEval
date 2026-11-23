# AIXeval

Post-hoc scorecard for a **saved** agent run: ingest a dump or trace,
score it against a 47-metric catalog, and read an eight-question card
(capability, robustness, safety, fairness, transparency, governance,
oversight, efficiency), each graded 0–4. Use the Streamlit UI, the
`aixeval` CLI, or the Python library. The toolkit does **not** execute
agents and is not a live safety control.

Python **≥ 3.12**. Package name: `aixeval`.

Latest release: [v0.1.1](https://github.com/VectorInstitute/AIXEval/releases/tag/v0.1.1) ·
DOI: [10.5281/zenodo.24468828](https://doi.org/10.5281/zenodo.24468828)

## Quick start

From a clone ([uv](https://docs.astral.sh/uv/) recommended):

```bash
uv sync --dev
uv run aixeval score fixtures/simple_match.json --out run.json
uv run aixeval report run.json
```

Streamlit UI (optional):

```bash
uv sync --dev --group ui
uv run --group ui streamlit run src/aixeval/ui/app.py
```

Copy [`.env.example`](.env.example) to `.env` when you use **live-judge**
metrics (LLM calls). See [Judge providers](docs/judge-providers.md).

## Documentation

| Topic | Where |
|--------|--------|
| Docs site map | [`docs/getting-started/index.md`](docs/getting-started/index.md) (Get started) |
| Install, UI, CLI, tests | [`docs/user_guide.md`](docs/user_guide.md) |
| Eight dimensions, bands, hard stops | [`docs/methodology.md`](docs/methodology.md) |
| Metric catalog (all 47 ids) | [`docs/metric-catalog.md`](docs/metric-catalog.md) |
| Add or change a metric | [`docs/extending/add-a-metric.md`](docs/extending/add-a-metric.md) |
| About, acknowledgments, citation | [`docs/about/index.md`](docs/about/index.md) |
| What a grade does *not* mean | [`docs/interpreting-results.md`](docs/interpreting-results.md) |
| Privacy and judge data flows | [`docs/privacy-and-data.md`](docs/privacy-and-data.md) |
| Self-hosted judge (vLLM) | [`docs/self-hosting.md`](docs/self-hosting.md) |

Docs site: <https://vectorinstitute.github.io/AIXEval/>. Preview it
locally with `uv run mkdocs serve`.

Per-metric rationale and citations also live in module docstrings under
`src/aixeval/scorecard/`.

## Library example

```python
from aixeval.ingest import ingest_dump
from aixeval.scorecard import build_profile, score_metrics
from pathlib import Path

record = ingest_dump(Path("fixtures/simple_match.json"))
card = build_profile(score_metrics(record))
```

Traces and overlays: `aixeval.ingest.pipeline.ingest_any`.

## Important limitations

- Not certification, compliance (including EU AI Act), penetration testing,
  or a substitute for domain safety testing.
- Not for decisions about people (hiring, medical care, legal outcomes).
- A high score does **not** mean a system is safe to deploy.

Do not score traces with personal, confidential, or regulated data without
authorization. Read [`docs/privacy-and-data.md`](docs/privacy-and-data.md)
before sending content to a hosted judge API.

## Contributing

[`CONTRIBUTING.md`](CONTRIBUTING.md) — tests, style, and pull requests.

## Licence

Apache License 2.0. Copyright © Vector Institute. See [`LICENSE.md`](LICENSE.md)
and [`NOTICE`](NOTICE). Vector names and logos are trademarks and are
**not** licensed under Apache.

## Acknowledgments

Resources used in preparing this research were provided, in part, by the
Province of Ontario, the Government of Canada through CIFAR, and
companies sponsoring the Vector Institute.

This research was funded by the European Union's Horizon Europe research
and innovation programme under the AIXPERT project (Grant Agreement
No. [101214389](https://cordis.europa.eu/project/id/101214389)). Views
and opinions expressed are those of the authors only and do not
necessarily reflect those of the European Union.

## Citation

See [`CITATION.cff`](CITATION.cff).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24468828.svg)](https://doi.org/10.5281/zenodo.24468828)

**APA (Zenodo):**

```text
Narayanan, A., & Raza, S. (2026). AIXeval (Version v0.1.1) [Computer software]. Zenodo.
https://doi.org/10.5281/zenodo.24468828
```

Source code: https://github.com/VectorInstitute/AIXEval
