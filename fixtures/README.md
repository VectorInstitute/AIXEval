# Fixtures

Synthetic dumps, traces, and demo scenarios for tests and the Streamlit
UI. They are not an evaluation corpus.

## Attestation

- **Hand-written examples** (everything except
  `scenarios/multidomain/`) were authored by the AIXeval team and
  reviewed for personal or sensitive information on **2026-11-11**.
- **`scenarios/multidomain/`** (22 episodes) was written by an LLM
  (`author_model` in each `scenario.json`) from task specs the AIXeval
  team wrote. Its tool sequences and pass/fail labels were set by
  code, not by the model. It was screened on **2026-11-20** for emails,
  URLs, phone numbers, and real institution names. Contact details use
  `example.com`.
- All examples are **fully fictional**. They are not derived from real
  cases, real applicants, real patients, real newsrooms, or real
  dockets. Names, employers, case captions, and medical details are
  invented.
- **TRAIL** (Patronus) and **τ-bench** data are **excluded**. Those
  datasets are not redistributed here (`fixtures/val_samples/` and
  `fixtures/taubench_samples/` are gitignored; `trail_data/` is
  gitignored). The Patronus-shaped file `patronus_nested_run.json` is
  a hand-written nest, not a TRAIL export. The multidomain traces use
  the τ-bench trace **format** (`task_id`, `traj`, `info`), and none
  of their content comes from τ-bench.

If you add a fixture, keep it fictional, keep PII out, and extend this
note.

## Layout

| Path | What it is |
|---|---|
| `*.json` at this level | Small ingest shapes (plain dump, Langfuse, OTel, Patronus-shaped, gold-in-input leak). |
| `scenarios/<domain>/<name>/` | One prefilled UI/CLI demo: `trace.json` plus optional `declared.json`, `overlay.json`, `scenario.json`, `saved_run.json`. |
| `scenarios/multidomain/<slug>/` | **22** frozen LLM-authored episodes in τ-bench trace format, across healthcare, legal, hiring, and education. See [multidomain/README.md](scenarios/multidomain/README.md). |

`saved_run.json` is the UI snapshot (`record` / `rows` / `rationale`),
**not** a run bundle. `aixeval report` will reject it. Score the
trace if you want a bundle:

```bash
uv run aixeval score fixtures/scenarios/hiring/resume_injection_followed/trace.json --out runs/demo.json
uv run aixeval report runs/demo.json
```

## Intentional negative examples

These scenarios **deliberately** show bad or incomplete agent
behaviour. They are test cases, not endorsements.

| Folder | Content note |
|---|---|
| `hiring/resume_injection_followed` | Hidden instruction in a résumé is obeyed; an unqualified applicant is advanced. |
| `journalism/outlet_brief_invented_disagreement` | A brief invents a disagreement that neither source contains. |
| `healthcare/f2_stereotype_screen` | Planted stereotypes in a public-health announcement; the screener misses one. |
| `healthcare/f2_stereotype_review_panel` | Same planted stereotypes, multi-agent review panel. |
| `healthcare/neighborhood_alerts_messy_overclaimed` | Fairness of a heat-alert policy is overclaimed from messy logs. |
| `hiring/rejection_email_unconfirmed` | A rejection email is sent without the required confirm. |
| `healthcare/lab_order_unconfirmed` | A lab order goes out without the required confirm. |
| `legal/citation_loop_stalls` | A citation loop retries without making progress. |

Paired “handled well” folders sit next to several of these (e.g.
`resume_injection_resisted`, `lab_order_confirmed`).

In `scenarios/multidomain/`, every episode whose `condition` is not
`success` is a negative example (14 of 22). The condition names the
flaw:

| Condition | Flaw shown |
|---|---|
| `missed_confirm` | A consequential write (grade, page, email, court filing) without the required confirmation. |
| `fairness_skew` | Equivalent people or items treated differently with no supported reason. |
| `policy_breach` | A hidden instruction in the input is obeyed. |
| `weak_grounding` | The answer states a fact no tool result supports. |
| `bloated` | Redundant tool calls and padded narration. |

Each of those `scenario.json` files sets `"intentional_failure": true`
and a `content_note`.

## Judge model on saved example grades

New **run bundles** (`schema_version: "1"`) record `judge` when live
metrics ran.

The shipped `saved_run.json` files **do not** record a `judge` field.
Nine of them name the judge model in their `source` text (for example
`claude-opus-5-5`); the rest do not name it at all. Do not treat those
grades as reproducible against a live provider. Re-score with `--live-judges`
(or the UI with replay off) if you need a named model on the card.
