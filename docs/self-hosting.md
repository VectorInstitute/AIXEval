# Self-hosting a judge

Run an open-weight model on your own GPU (or an on-prem cluster) and
point AIXeval at it with `JUDGE_PROVIDER=vllm`. Prompts and traces then
stay on infrastructure you control.

This is the path to use when you cannot send run content to a hosted
API. Read [Privacy and data](privacy-and-data.md) first.

## What you need

- A machine the toolkit can reach over HTTP (same host is fine).
- [vLLM](https://docs.vllm.ai/) (or any OpenAI-compatible chat server)
  serving the model you want as judge.
- In `.env`:

```bash
JUDGE_PROVIDER=vllm
JUDGE_MODEL_VLLM=your-served-model-name
VLLM_BASE_URL=http://localhost:8000/v1
# VLLM_API_KEY=   # only if you started the server with --api-key
```

`JUDGE_MODEL_VLLM` has **no default**. It must match `--served-model-name`.

## Start to finish

The full install, `vllm serve` flags, and empty-reply/`finish_reason`
gotchas are in [Self-hosted vLLM judge](judge-vllm.md).

## Check it

```bash
uv run python -m aixeval.judge.config
uv run aixeval score fixtures/simple_match.json --live-judges grounding --out runs/local_judge.json
uv run aixeval report runs/local_judge.json
```

The card and the bundle should name your served model. If the server
stalls or hits a token cap, the error should mention `finish_reason`.

## What this does not do

Self-hosting the **judge** does not self-host the agent under test.
It only keeps *evaluation* prompts off a vendor API. Computed metrics
already run locally and do not need vLLM.
