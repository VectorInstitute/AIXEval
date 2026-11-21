# Contributing

This repo is a post-hoc **scorecard** (`aixeval`). Ingest, scorecard,
judge, CLI, and Streamlit UI are in tree. `harness/` and `serve/` are
stubs.

Setup and CLI/UI: [README.md](README.md), [docs/user_guide.md](docs/user_guide.md).
New metrics: [docs/extending/add-a-metric.md](docs/extending/add-a-metric.md).
Metric ids: [docs/metric-catalog.md](docs/metric-catalog.md). Do not send traces with personal or regulated data to a hosted judge
without authorization; see [docs/privacy-and-data.md](docs/privacy-and-data.md).

## Support

Vector AI Engineering (`shaina.raza@vectorinstitute.ai`) triages
GitHub issues.
Security reports: [SECURITY.md](SECURITY.md) — do not file those as
public issues.

## Pull requests

Fill out `.github/pull_request_template.md`. Link the issue if there is one.

Do not commit `.env`, local run files, or secrets. Do not add
`.agents/memory/`, `.agents/decisions/`, `AGENTS.md`, or `.cursor/rules/`
unless a maintainer asks.

## Dry run before you ask for review

```bash
uv sync --dev
source .venv/bin/activate
uv run pytest tests/
uv run mypy src
uv run ruff check .
uv run pre-commit run --all-files
```

This checks scorecard, ingest, judge, and UI-logic unit tests. It does
not pull live traces. Three tests are marked `integration_test` and skip
if no provider key is in `.env` — see [user_guide.md](docs/user_guide.md)
deliberately. Streamlit AppTests need `uv sync --group ui`.

## Style

- Python `>=3.12`, `src/` layout, `uv` for dependencies
- PEP 8; NumPy docstrings; types required (`ruff` + `mypy`)
- Line length 120 (ruff)
- Tests: pytest. Mark live-provider tests `@pytest.mark.integration_test`
- Prefer `uv run` so tools use the project environment
