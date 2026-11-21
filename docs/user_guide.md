# User guide

AIXeval scores a **saved** agent run. It does not execute the agent.
Ingest a dump or trace, score a 47-metric catalog, and read an
eight-question card. The [metric catalog](metric-catalog.md) lists every
metric id and required fields.

Python `>=3.12`. Dependencies are managed with
[uv](https://docs.astral.sh/uv/).

## Install

```bash
uv sync --dev
source .venv/bin/activate
```

`uv run <cmd>` works without activate. Extra groups:

| Group | What it adds |
|---|---|
| `dev` | pytest, ruff, mypy, pre-commit (default with `--dev`) |
| `docs` | mkdocs (default in `uv sync`) |
| `ui` | Streamlit: needed to run the app and `tests/test_ui_app.py` |

```bash
uv sync --dev --group ui
```

The opt-in `meteor` metric needs a one-time NLTK download. The test
suite does not:

```bash
uv run python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Streamlit UI

```bash
uv run --group ui streamlit run src/aixeval/ui/app.py
```

Run it from the repo root so `.streamlit/config.toml` is picked up. That
config binds the UI to `localhost` only. On a remote or shared machine,
reach it through an SSH tunnel (`ssh -L 8501:localhost:8501 <host>`)
rather than exposing the port: anyone who can open the UI can run judge
calls on your API keys.

Upload a JSON dump, Langfuse/OTel/Patronus trace, or Inspect AI `.eval`
file. Optionally declare gold fields and a multi-agent overlay, pick
metrics, and run. The sidebar **Judge** control sets the live provider
and model for this browser session; it does not write `.env`.

Live judge setup: [self-hosting](self-hosting.md),
[provider keys](judge-providers.md), [self-hosted vLLM](judge-vllm.md).

Read [methodology](methodology.md), [interpreting results](interpreting-results.md),
[limitations](limitations.md), and [privacy](privacy-and-data.md) before
you score anything that is not a fixture.

## CLI

```bash
aixeval score dump.json --out run.json
aixeval report run.json
aixeval replay run.json
aixeval compare run_a.json run_b.json
```

`score --live-judges grounding,refusal` needs a configured judge
provider. `replay` re-scores the saved record with no model calls.

## Docs site

```bash
uv run mkdocs serve
```

Opens on `http://127.0.0.1:8000`. The [API reference](api.md) is
generated from the package.

## Tests

```bash
uv run pytest -m "not integration_test"   # default; no API key
uv run pytest -m "integration_test"       # 3 live tests; need .env
uv run mypy src
uv run ruff check .
uv run pre-commit run --all-files
```

Three tests are marked `integration_test`: one ingest→judge→score path
(`tests/test_judge_integration.py`) and two cross-modal image checks
(`tests/test_judge_cross_modal_hallucination_image_integration.py`).
Each skips if no provider key is set.

Layers, grains, and hard stops: [Methodology](methodology.md).
