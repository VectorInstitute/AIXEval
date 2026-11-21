# Self-hosted vLLM judge

Use this guide if you run your own [vLLM](https://docs.vllm.ai/) server and
want AIXeval to call it as the live judge.

It talks to **your** OpenAI-compatible endpoint.

## What the client needs

| Setting | Env var | Required? |
|---|---|---|
| Provider name | `JUDGE_PROVIDER=vllm` | Yes, to select this client |
| Model id | `JUDGE_MODEL_VLLM` | Yes. There is no default: it must match `--served-model-name` (or the weights path) |
| Server URL | `VLLM_BASE_URL` | No. Default `http://localhost:8000/v1` |
| API key | `VLLM_API_KEY` | Only if you started vLLM with `--api-key` |

Copy [`.env.example`](https://github.com/VectorInstitute/aixeval/blob/main/.env.example)
and fill those in. Switching `JUDGE_PROVIDER` does not erase the other
providers' model settings.

The Python client is `aixeval.judge.vllm_client.VLLMClient`. It uses the
OpenAI Chat Completions API (`chat.completions.create`), not the newer
Responses API.

## 1. Install vLLM (its own environment)

Do not add `vllm` to this project's `uv` environment. It pins its own
PyTorch. Create a separate venv:

```bash
uv venv "$HOME/envs/vllm" --python 3.12
source "$HOME/envs/vllm/bin/activate"
uv pip install vllm
python -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.version.cuda)"
```

The printed CUDA version must be one your GPU driver supports.

## 2. Start the server

Point `vllm serve` at a local Hugging Face-format folder, or at a Hub repo
id you have already downloaded.

```bash
# Optional: only if other machines will reach this port
openssl rand -hex 16 > ~/.vllm_key && chmod 600 ~/.vllm_key

vllm serve /path/to/gemma-3-12b-it \
  --served-model-name gemma-3-12b-it \
  --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --api-key "$(cat ~/.vllm_key)"
```

`--served-model-name` is the string you put in `JUDGE_MODEL_VLLM`.
`--host 127.0.0.1` is enough if the toolkit runs on the same machine;
use `0.0.0.0` only if another host must connect, and then set `--api-key`.

Wait until the log prints `Application startup complete`. Then:

```bash
curl -s http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $(cat ~/.vllm_key)"
```

You should see `gemma-3-12b-it` (or whatever name you served).

### GPU size (rough)

| Checkpoint | Memory (bf16) | Typical card |
|---|---|---|
| ~8B | ~16 GB | one 24 GB GPU |
| ~12B (Gemma 3 12B) | ~24 GB | one 48 GB GPU (A40) |
| ~27B | ~54 GB | two 48 GB GPUs (`--tensor-parallel-size 2`) or one 80 GB GPU |

A vision checkpoint (Gemma 3, Qwen2.5-VL, LLaVA) is required for
`complete_with_image` (the cross-modal hallucination judge). A text-only
model errors on that call instead of silently skipping it.

## 3. Point AIXeval at the server

In the repo `.env`:

```bash
JUDGE_PROVIDER=vllm
JUDGE_MODEL_VLLM=gemma-3-12b-it
VLLM_BASE_URL=http://127.0.0.1:8000/v1
VLLM_API_KEY=   # paste the key if you passed --api-key
```

If the server is on another host, use that host's name or IP in
`VLLM_BASE_URL`. The path must end in `/v1`.

## 4. Check the client

From the AIXeval repo:

```bash
uv run python -m aixeval.judge.vllm_client
```

Expect a one-word `ok`. Then either:

```bash
uv run python -m aixeval.judge.grounding
```

or, in the Streamlit app (`uv run --group ui streamlit run src/aixeval/ui/app.py`),
set **Provider** to `vllm` in the sidebar, click **Test connection**, and run
an evaluation with replay off.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Set JUDGE_MODEL_VLLM ... there is no default` | The model env var is empty |
| `Failed to infer device type` / NVML errors | You started vLLM on a machine with no GPU |
| `Vector/vLLM reply had no text` and `finish_reason='length'` | A reasoning model spent the token budget on hidden thinking. Raise `max_tokens`. The sidebar test already uses 2048 |
| Image judge fails, text judges work | The served checkpoint is text-only |
| Connection refused | Wrong host/port, or the server has not finished loading weights |
| 401 Unauthorized | Server was started with `--api-key` and `VLLM_API_KEY` does not match |

If FlashInfer tries to compile CUDA kernels and dies with `Could not find nvcc`,
the node has no CUDA toolkit. Start the server with:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

That uses vLLM's built-in sampler. It is slower, not incorrect.
