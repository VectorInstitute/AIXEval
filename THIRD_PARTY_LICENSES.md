# Third-party licences

Python dependencies and their licences are listed by
[pip-licenses](https://pypi.org/project/pip-licenses/) from a
resolved environment:

```bash
uv run pip-licenses --from=mixed --format=markdown --output-file THIRD_PARTY_LICENSES.generated.md
```

The generated dump is not committed (it churns with every lock bump).
Regenerate it before a release and attach it to the GitHub Release if
legal review wants the full table.

This file records **runtime-downloaded models** and **fonts**, which
pip-licenses does not see.

## Runtime-downloaded models

These are **not** vendored. They download on first use of the opt-in
metric.

| Artifact | Used by | Source | Licence (upstream) |
|---|---|---|---|
| `unitary/toxic-bert` | opt-in `toxicity` classifier path | Hugging Face | Apache-2.0 (Unitary) |
| DeBERTa weights pulled by `bert-score` | opt-in `bertscore` | Hugging Face / `bert-score` default | MIT (`bert-score`); model card of the pulled checkpoint |
| `openai/clip-vit-base-patch32` | opt-in `clip_score` | Hugging Face | MIT (OpenAI CLIP) |

Do not call those metrics if you cannot accept the download and the
upstream licence.

## Fonts

The Streamlit theme and MkDocs site load **Open Sans** and
**Montserrat** from Google Fonts. Both families are licensed under
the SIL Open Font License (OFL). They are not bundled as binary font
files in this repository.

Vector logos in `docs/assets/` are trademarks; see `NOTICE`. They are
not OFL and not Apache-2.0 licensed for reuse.

## Judge providers

Hosted SDKs (`anthropic`, `openai`, `google-genai`) are ordinary
PyPI dependencies. Using them sends prompts to that vendor under
**their** terms, not this licence. Self-hosted vLLM is a separate
install; this repo does not vendor vLLM.
