# Judge providers

`JUDGE_PROVIDER` in `.env` selects the default client. The Streamlit
sidebar can switch provider and model for **this browser session**
without writing `.env`.

Copy [`.env.example`](https://github.com/VectorInstitute/AIXEval/blob/main/.env.example).
Only the selected provider's key is required.

## Hosted APIs (data leaves your machine)

| `JUDGE_PROVIDER` | Key | Model env var | Default model |
|---|---|---|---|
| `anthropic` (default) | `ANTHROPIC_API_KEY` | `JUDGE_MODEL_ANTHROPIC` | `claude-sonnet-5` |
| `openai` | `OPENAI_API_KEY` | `JUDGE_MODEL_OPENAI` | `gpt-5.5` |
| `gemini` | `GEMINI_API_KEY` | `JUDGE_MODEL_GEMINI` | `gemini-flash-latest` |

These send live-judge prompts to that vendor. See
[Privacy and data](privacy-and-data.md).

## Self-hosted (data stays on your server)

| `JUDGE_PROVIDER` | Key | Model env var | Default model |
|---|---|---|---|
| `vllm` | `VLLM_API_KEY` only if the server was started with `--api-key` | `JUDGE_MODEL_VLLM` (**required**, no default) |: |

Optional: `VLLM_BASE_URL` (default `http://localhost:8000/v1`).

Guide: [Self-hosting](self-hosting.md) → [vLLM howto](judge-vllm.md).

## Switching mid-session

The UI sidebar **Judge** control changes provider/model for the next
run. Replay a scenario to reuse `saved_run.json` verdicts; turn replay
off to call the sidebar judge.

The CLI uses `.env` (or the process environment) only:

```bash
JUDGE_PROVIDER=vllm JUDGE_MODEL_VLLM=Qwen2.5-7B-Instruct \
  uv run aixeval score fixtures/simple_match.json --live-judges grounding
```
