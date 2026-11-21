# Roadmap

AIXeval ships **post-hoc** evaluation first. Live **harness** and hosted
**serve** capabilities are **Phase 2** (planned, not in this release).

## Phase 1 (available now)

| Area | What you get |
|---|---|
| `scorecard/` | 47 metrics, grading, eight-question card |
| `ingest/` | Dumps, traces, declared overlay |
| `judge/` | LLM judges (API or self-hosted vLLM) for metric calls |
| `ui/` | Streamlit scorecard over finished runs |
| `cli` | `aixeval score` / `report` on saved JSON |

Phase 1 answers: **given a run that already finished**, how does it score
on the catalog and the trustworthiness card?

## Phase 2 (incoming)

| Area | Direction (not implemented) |
|---|---|
| `harness/` | **Live evaluation harness**: drive or wrap agent execution so
  runs can be produced, gated, or interrupted under policy (for example
  confirmation before high-impact actions, scripted scenarios, repeatable
  eval loops). Today `harness/` is a **stub** only. |
| `serve/` | **Hosted evaluation service**: auth, queueing, and storage so
  teams can submit traces and retrieve scorecards without running the UI
  locally. Today `serve/` is a **stub** only. |

Phase 2 does **not** change the meaning of Phase 1 metrics. It extends
**how** runs are produced and **how** the toolkit is operated at scale.

Package layout keeps `harness/` and `serve/` as placeholders so public
APIs and docs can grow without a breaking rename later.

## What to use today

- Finished run already? Start with the [user guide](../user_guide.md) or
  [Get started](../getting-started/index.md).
- Need confirmation-gating or live-path controls? Not in this repo yet;
  track Phase 2 above or integrate your own runner and score with Phase 1.

Questions: open a GitHub issue or contact
`shaina.raza@vectorinstitute.ai`.
