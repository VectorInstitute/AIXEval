# Multidomain LLM-authored demos

Twenty-two frozen episodes across healthcare, legal, hiring, and
education, written by an LLM (`author_model` in each `scenario.json`).
Code fixed each episode's tool sequence, confirmation turns, and
pass/fail labels; the model wrote the text. All entities are fictional.

Each folder has the usual scenario layout (`trace.json`,
`declared.json`, `scenario.json`). There is no `saved_run.json`, so
score them live or with replay off.

- `condition` in `scenario.json` names the variant: `success`, or one
  of the flaws listed in [fixtures/README.md](../../README.md#intentional-negative-examples).
- Negative examples set `"intentional_failure": true` and a
  `content_note`.
- `manifest.json` lists the episodes. The UI and CLI ignore it when
  discovering scenarios.
