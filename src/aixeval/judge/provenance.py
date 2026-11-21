"""Per-verdict judge identity written onto the sealed record.

Runners already keep ``model`` / ``prompt_version`` on a side
attachment. The live path used to write only the score. Both now
also merge a ``judge_provenance`` map onto the record so a replayed
verdict can name the model that produced it.
"""

from __future__ import annotations

from typing import Any

from aixeval.judge.provider import Provider


PROVENANCE_FIELD = "judge_provenance"


def provenance_entry(client: Provider, prompt_version: str) -> dict[str, str]:
    """One metric's provenance: the model id and the prompt that produced it."""
    return {"model": client.model, "prompt_version": prompt_version}


def merge_judge_provenance(
    record: dict[str, Any], metric_id: str, client: Provider, prompt_version: str
) -> dict[str, Any]:
    """Return a copy of ``record`` with this metric's provenance set.

    Existing entries for other metrics are kept. An existing entry for
    ``metric_id`` is replaced. The input dict is not mutated.
    """
    updated = dict(record)
    provenance = dict(updated.get(PROVENANCE_FIELD) or {})
    provenance[metric_id] = provenance_entry(client, prompt_version)
    updated[PROVENANCE_FIELD] = provenance
    return updated


def models_from_provenance(record: dict[str, Any] | None) -> tuple[str, ...]:
    """Distinct model ids recorded on ``record``, in first-seen order."""
    if not record:
        return ()
    raw = record.get(PROVENANCE_FIELD)
    if not isinstance(raw, dict):
        return ()
    models: list[str] = []
    seen: set[str] = set()
    for entry in raw.values():
        if not isinstance(entry, dict):
            continue
        model = entry.get("model")
        if isinstance(model, str) and model and model not in seen:
            seen.add(model)
            models.append(model)
    return tuple(models)
