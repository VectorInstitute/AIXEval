# Add a metric

This guide is for **first-party** metrics merged into `aixeval`. Follow
the steps in order; the test suite enforces that registries stay in sync.

## Scorer contract

Each metric lives in `src/aixeval/scorecard/<metric_id>.py` and exposes:

- A module-level `METRIC_ID = "<metric_id>"` string (snake_case).
- `score_<metric_id>(record: dict[str, Any]) -> dict[str, Any]`.

Return shape:

| `status` | Fields |
|---|---|
| `scored` | `metric_id`, `raw_score` on **[0, 1]**, `grade` (0–4), `band_label` |
| `na` | `metric_id`, `na_reason` (use `require_na_reason` from `na_reasons.py`) |

Use `apply_grade()` from `profile.py` for scored rows so bands match the
rest of the catalog. Missing evidence must be **N/A**, never a silent `0.0`.

Study an existing scorer:

- **Computed / library:** [`bleu.py`](https://github.com/VectorInstitute/aixeval/blob/main/src/aixeval/scorecard/bleu.py)
- **Token overlap:** [`f1.py`](https://github.com/VectorInstitute/aixeval/blob/main/src/aixeval/scorecard/f1.py)
- **Declared bools:** see `error_recovery.py`, `unsafe_action.py`

Document literature, edge cases, and required record fields in the module
docstring. The [metric catalog](../metric-catalog.md) table is a summary;
the module is the detailed source until per-metric pages are generated.

## Registration checklist

Complete every row that applies to your metric.

| Step | File | What to add |
|---|---|---|
| 1 | `src/aixeval/scorecard/<metric_id>.py` | Scorer + tests in `tests/test_<metric_id>.py` (or named file) |
| 2 | `src/aixeval/scorecard/metrics.py` | Id in `POSTHOC_METRICS` **or** `OPT_IN_METRICS` |
| 3 | `metrics.py` | `METRIC_DIMENSION[<id>]`: one of the eight card questions |
| 4 | `metrics.py` | Import + call inside `score_metrics()` (skip if opt-in) |
| 5 | `definitions.py` | One sentence in `METRIC_DEFINITIONS` or `OPT_IN_METRIC_DEFINITIONS` |
| 6 | `sources.py` | Row in `METRIC_SOURCES` (`kind`, `citation`, `note`) |
| 7 | `reband.py` | If **lower-is-better**, list id in `LOWER_IS_BETTER_METRICS` |
| 8 | `profile.py` | Only if policy needs it: new `HARD_STOP_RULES` entry (rare) |
| 9 | `docs/metric-catalog.md` | Summary row until catalog generation lands (Phase 2) |
| 10 | Optional deps | `pyproject.toml` via `uv add`; document in user guide if heavy |

### POSTHOC vs opt-in

**POSTHOC** ids run automatically in `score_metrics()` and appear in default
UI flows. Put a metric in **OPT_IN** when:

- It needs large downloads or GPU libs (e.g. `bertscore`, `clip_score`).
- Automatic inclusion would distort the card (e.g. `meteor` on short exact
  answers: see the comment at the top of `metrics.py`).

Opt-in metrics still need definitions for `ALL_METRIC_IDS` and the judge
suggester; callers invoke `score_<id>(record)` directly.

### Lower-is-better

Call `apply_grade(..., higher_is_better=False)` and add the id to
`LOWER_IS_BETTER_METRICS`. `tests/test_reband.py` scans scorer source and
fails if the set drifts.

### Hard stops

Hard stops (**contamination**, **unsafe action**, **confirmation gating**)
only change **card** questions; they never change `raw_score`. New metrics
almost never add hard stops. If you believe one is required, discuss in the
PR and update `HARD_STOP_RULES` in `profile.py` plus
`docs/methodology.md`.

## Tests that guard the catalog

These tests fail when registries drift:

- `tests/test_definitions.py`: every `POSTHOC_METRICS` / opt-in id has a definition
- `tests/test_metric_sources.py`: `METRIC_SOURCES` covers all posthoc ids
- `tests/test_metric_dimensions.py`: `METRIC_DIMENSION` keys match posthoc set
- `tests/test_score_metrics.py`: one row per posthoc id
- `tests/test_reband.py`: lower-is-better set matches scorer code

Run before review:

```bash
uv sync --dev
uv run pytest tests/test_<your_metric>.py tests/test_definitions.py tests/test_metric_sources.py
uv run mypy src
uv run ruff check .
```

## Private metrics without upstream merge

You do not have to fork `score_metrics()` for experiments:

1. Implement `score_my_metric(record)` with the same return dict shape.
2. After ingest, call it alongside `score_metrics(record)`.
3. Pass the combined list to `build_profile()` **only if** you assigned
   `METRIC_DIMENSION`-compatible behavior and accept card aggregation rules
   (each question grade is derived from metrics mapped to that dimension).

Rows with unknown `metric_id` values are not folded into the card unless
you extend profile logic upstream. Keeping a **sidecar JSON** report is fine
for org-only metrics.

## Pull request

Fill out `.github/pull_request_template.md`. Link an issue if one exists.
See [Contributing](contributing.md) for style and the full dry-run list.

In the PR description, state:

- Mechanism (declared / computed / judge / overlay)
- Required record fields (ingest contract)
- Why POSTHOC vs opt-in
- Any new optional dependency and license impact