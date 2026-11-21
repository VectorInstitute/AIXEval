# Contributing

The repository is a post-hoc **scorecard** (`aixeval`): ingest, scorecard,
judge, CLI, and Streamlit UI. `harness/` and `serve/` are Phase 2 stubs (see
[Roadmap](../about/roadmap.md)).

| Task | Where to start |
|---|---|
| Install, UI, CLI | [User guide](../user_guide.md) |
| Add or change a metric | [Add a metric](add-a-metric.md) |
| Metric ids and fields | [Metric catalog](../metric-catalog.md) |
| Judge data handling | [Privacy and data](../privacy-and-data.md) |

The canonical copy of this page for GitHub also lives at
[`CONTRIBUTING.md`](https://github.com/VectorInstitute/aixeval/blob/main/CONTRIBUTING.md)
in the repository root.

## Support

Vector AI Engineering (`shaina.raza@vectorinstitute.ai`) triages GitHub
issues.

Security reports: [`SECURITY.md`](https://github.com/VectorInstitute/aixeval/blob/main/SECURITY.md).
Do not file those as public issues.

## Pull requests

Use the pull request template in the repo. Link the issue when there is one.

Do not commit `.env`, local run files, or secrets.

## Dry run before review

```bash
uv sync --dev
source .venv/bin/activate
uv run pytest tests/
uv run mypy src
uv run ruff check .
uv run pre-commit run --all-files
```

Unit tests cover scorecard, ingest, judge, and UI logic without live traces.
Tests marked `integration_test` skip when no provider key is in `.env`;
see the [user guide](../user_guide.md). Streamlit AppTests need
`uv sync --group ui`.

## Style

- Public docs (`docs/`): do not use em dashes (Unicode U+2014). Use commas,
  colons, parentheses, or semicolons instead.
- Python `>=3.12`, `src/` layout, `uv` for dependencies
- PEP 8; NumPy docstrings; types required (`ruff` + `mypy`)
- Line length 120 (ruff)
- Tests: pytest; mark live-provider tests `@pytest.mark.integration_test`
- Prefer `uv run` so tools use the project environment