# Privacy and data

Do **not** evaluate traces that contain personal, confidential, or
regulated information unless you have authorization to process that
information in this toolkit, including any judge provider you select.

If that bar is not met, do not upload the trace. Use a
[self-hosted judge](self-hosting.md) or computed-only metrics on
authorized machines.

## What stays on the machine

In the Streamlit UI:

- Uploaded traces live in **session memory**. They are not written to
  the repo or to an AIXeval server.
- Results are written to disk **only** when you click **Download run
  bundle**. Closing the tab drops the session.
- This toolkit does not log trace content.

The CLI writes a bundle only when you pass `--out`. `report`, `reband`,
`replay`, and `compare` read the files you point at; they do not upload
them.

Streamlit usage telemetry is turned off in `.streamlit/config.toml`
(`gatherUsageStats = false`).

## What leaves the machine

A metric sends data to a **provider** only when it is a live-judge
metric (or the metric suggester) **and** you actually run it.

| Path | Data that can leave the machine | Destination |
|---|---|---|
| Live-judge metrics listed below | Output, plus evidence/trajectory/image as that metric requires | The configured judge provider |
| UI “ask an LLM which metrics are relevant” | Short run facts (source kind, agent count, image present, …) and your description | The configured judge provider |
| Computed / declared / overlay metrics | None | Local process |
| Opt-in `bertscore` / `meteor` / `clip_score` / local `toxic-bert` | None (after the one-time model download from Hugging Face) | Local process |

Live-judge metric ids: `grounding`, `trajectory_optimality`,
`tool_call_accuracy`, `abstention`, `refusal`, `toxicity`,
`pairwise_preference`, `cross_modal_hallucination`,
`protected_attribute_bias`.

Providers:

- `anthropic`, `openai`, `gemini`: those vendors' APIs.
- `vllm`: **your** OpenAI-compatible server. If that server is on
  your network, the prompt stays in-house.

See [Judge providers](judge-providers.md) for keys and env vars.

## Authorization warning

Do not evaluate traces with personal, confidential, or regulated
information without authorization. That includes traces that contain
health records, applicant data, case files, or anything you would not
paste into the judge provider you selected.

If you cannot authorize a hosted API, run a
[self-hosted open-weight judge](self-hosting.md) so prompts never
leave your facility, or score only local metrics.

## Temp files

An uploaded reference image is copied to the OS temp directory for the
cross-modal judge. See [Limitations](limitations.md#reference-images-may-remain-in-the-os-temp-folder).
