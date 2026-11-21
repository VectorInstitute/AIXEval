"""Export one fully-scored run as a single portable, versioned JSON bundle.

Today a scored run only exists as whatever is currently in a Streamlit
session or a caller's own variables; there is no way to save a
complete run (the sealed record, its trace, every metric's result, and
the eight-question profile) to reopen, share, or diff later. This is
the useful core of an older, parked design's "portable zip bundle" idea
(``feature/portable-eval-kernel``, dormant, shares only the repo's root
commit with this branch), kept narrow on purpose: a flat JSON export
of what ``ingest_any`` / ``score_metrics`` / ``build_profile`` already
produce, not a new live-capture schema or a scheduler. A full live-
harness Evidence Package stays roadmap, documented separately outside
this codebase; ``aixeval.cli`` wraps this module as
``aixeval score|report|reband|replay``.

``build_run_bundle`` takes the three pieces of output a caller already
has in hand after running the pipeline once: no new computation, no
model calls, nothing re-derived. ``replay_bundle`` is the inverse
check: given a saved bundle, confirm that re-scoring its own record
reproduces exactly what was saved, with no model call either.
``compare_bundles`` is a different check again: given *two*
independently-produced bundles, report whether their profiles
diverge beyond a tolerance, software support for an independent-rerun
safeguard, not a replacement for it.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from aixeval.ingest.pipeline import IngestResult


BUNDLE_SCHEMA_VERSION = "1"


def build_run_bundle(
    result: IngestResult,
    scored: list[dict[str, Any]],
    profile: dict[str, Any],
    *,
    judge: str | None = None,
) -> dict[str, Any]:
    """Assemble one JSON-serializable bundle for a fully-scored run.

    Parameters
    ----------
    result
        The ``IngestResult`` from ``ingest_any`` (or ``ingest_inspect_eval``'s
        per-epoch results). ``spans``, if present, are flattened to plain
        dicts; the bundle never holds live ``Span`` objects.
    scored
        ``score_metrics(result.record)``'s return value: one row per
        metric, already graded.
    profile
        ``build_profile(scored)``'s return value: the eight-question
        card plus ``applied_hard_stops``.
    judge
        Which judge model(s) produced the record's verdicts, e.g.
        ``"vllm · gemma-3-12b-it"``. Added as a ``judge`` key only when
        given, so bundles from callers that don't track it are unchanged.

    Returns
    -------
    dict
        A flat dict safe to pass straight to ``json.dumps``. Carries a
        ``schema_version`` so a future format change can be detected on
        read rather than silently misread.
    """
    bundle: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "source_kind": result.source_kind,
        "record": result.record,
        "spans": [asdict(span) for span in result.spans] if result.spans is not None else None,
        "trajectory": result.trajectory,
        "contamination_evidence": result.contamination_evidence,
        "scored_results": scored,
        "profile": profile,
    }
    if judge is not None:
        bundle["judge"] = judge
    return bundle


def write_run_bundle(bundle: dict[str, Any], path: str | Path) -> None:
    """Write a bundle built by ``build_run_bundle`` to ``path`` as formatted JSON.

    Creates parent directories if they do not exist, so ``--out runs/bundle.json``
    does not require a prior ``mkdir``.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")


def replay_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Re-score a bundle's own record and confirm it reproduces the saved results.

    No model call: every scorer prefers a verdict already written on the
    record over computing a fresh one (the same discipline ``grounding``,
    ``tool_call_accuracy``, and every other judge-backed metric already
    use), so re-running ``score_metrics`` on the bundle's own ``record``
    reproduces the same verdicts without calling a judge again. This is
    the offline-replay guarantee the Deliverable 2 design names as a
    Phase 0 exit criterion ("a golden Evidence Package replays bit-for-
    bit"), proven here for what this toolkit's own bundle format
    actually carries, not yet for a live harness's larger Evidence
    Package.

    Returns
    -------
    dict
        ``{"matches": bool, "metric_diffs": [...], "profile_diffs": [...]}``.
        Both diff lists are empty on a clean replay. Each diff entry
        names what was saved and what replay actually produced, so a
        mismatch is diagnosable, not just a bare failure.
    """
    from aixeval.scorecard.metrics import score_metrics  # noqa: PLC0415
    from aixeval.scorecard.profile import build_profile  # noqa: PLC0415

    replayed_all = score_metrics(bundle["record"])
    saved_by_id = {row["metric_id"]: row for row in bundle["scored_results"]}
    replayed_by_id = {row["metric_id"]: row for row in replayed_all}
    # Bundles may store a selected subset (CLI ``--metrics`` / scenario.json).
    # Replay the saved ids only: extra catalog rows are not part of this card.
    metric_diffs = [
        {"metric_id": metric_id, "saved": saved_by_id.get(metric_id), "replayed": replayed_by_id.get(metric_id)}
        for metric_id in sorted(saved_by_id)
        if saved_by_id.get(metric_id) != replayed_by_id.get(metric_id)
    ]
    replayed_results = [replayed_by_id[metric_id] for metric_id in saved_by_id if metric_id in replayed_by_id]
    replayed_profile = build_profile(replayed_results)

    saved_profile = bundle["profile"]
    profile_diffs = [
        {"dimension": key, "saved": saved_profile.get(key), "replayed": replayed_profile.get(key)}
        for key in sorted(set(saved_profile) | set(replayed_profile))
        if not _profile_values_match(key, saved_profile.get(key), replayed_profile.get(key))
    ]

    return {
        "matches": not metric_diffs and not profile_diffs,
        "metric_diffs": metric_diffs,
        "profile_diffs": profile_diffs,
    }


def compare_bundles(bundle_a: dict[str, Any], bundle_b: dict[str, Any], *, max_grade_drift: int = 1) -> dict[str, Any]:
    """Compare two independently-produced bundles of the same task.

    An independent-rerun check: re-executing the same task separately
    and flagging a profile that diverges too far, not a strict
    bit-for-bit match, since decoding/judge non-determinism across a
    real re-run is expected. This toolkit has no bootstrapped
    confidence interval to compare against, so ``max_grade_drift``
    (a plain band-count tolerance) is used in its place. This function
    does not re-execute anything
    itself; it only diffs two bundles a caller already produced,
    e.g. by running ``aixeval score`` twice (independently, or
    with live judges) on what is meant to be the same task.

    Parameters
    ----------
    bundle_a, bundle_b
        Two bundles from ``read_run_bundle``/``build_run_bundle``,
        ideally scoring the same underlying task.
    max_grade_drift
        How many grade bands apart two dimensions may be before
        being flagged as diverged. 1 (the default) tolerates the
        ordinary judge noise already documented for this toolkit's
        own judges (``trajectory_optimality``, ``pairwise_preference``);
        0 requires an exact grade match.

    Returns
    -------
    dict
        ``diverged`` (bool), and ``dimension_diffs``: one entry per
        dimension that diverges. A dimension where both sides have a
        grade and they differ by more than ``max_grade_drift`` is
        flagged with ``drift`` set to that gap. A dimension that is
        N/A on one side and scored on the other is also flagged
        (``drift`` is ``None``): missing evidence vs a real grade is
        a difference, not a match. Both-N/A is skipped.
    """
    profile_a, profile_b = bundle_a["profile"], bundle_b["profile"]
    dimension_diffs = []
    for dimension in sorted(set(profile_a) | set(profile_b)):
        if dimension == "applied_hard_stops":
            continue
        grade_a, grade_b = profile_a.get(dimension), profile_b.get(dimension)
        if grade_a is None and grade_b is None:
            continue
        if grade_a is None or grade_b is None:
            dimension_diffs.append({"dimension": dimension, "grade_a": grade_a, "grade_b": grade_b, "drift": None})
            continue
        drift = abs(grade_a - grade_b)
        if drift > max_grade_drift:
            dimension_diffs.append({"dimension": dimension, "grade_a": grade_a, "grade_b": grade_b, "drift": drift})
    return {"diverged": bool(dimension_diffs), "dimension_diffs": dimension_diffs}


def _profile_values_match(key: str, saved: Any, replayed: Any) -> bool:
    """Compare one profile field, tolerating JSON's tuple-becomes-list round-trip."""
    if key == "applied_hard_stops":
        return list(saved or ()) == list(replayed or ())
    return bool(saved == replayed)


def read_run_bundle(path: str | Path) -> dict[str, Any]:
    """Read back a bundle written by ``write_run_bundle``.

    Raises
    ------
    ValueError
        The file's ``schema_version`` is missing or doesn't match
        ``BUNDLE_SCHEMA_VERSION``: a newer or malformed bundle is
        rejected rather than silently misread.
    """
    path = Path(path)
    bundle = json.loads(path.read_text(encoding="utf-8"))
    version = bundle.get("schema_version")
    if version != BUNDLE_SCHEMA_VERSION:
        if _looks_like_ui_saved_run(bundle):
            msg = (
                f"{path} looks like a UI saved_run.json, not a run bundle. "
                "Score the scenario's trace and write a bundle first: "
                "aixeval score <folder>/trace.json --out run.json && aixeval report run.json"
            )
            raise ValueError(msg)
        msg = f"unknown run-bundle schema_version {version!r}, expected {BUNDLE_SCHEMA_VERSION!r}"
        raise ValueError(msg)
    return bundle


def _looks_like_ui_saved_run(payload: dict[str, Any]) -> bool:
    """UI scenario snapshots use ``rows``/``rationale``, not the bundle schema."""
    return "scored_results" not in payload and ("rows" in payload or "rationale" in payload)


def print_bundle_example() -> None:
    """Print a small round-tripped bundle: build, write, read back."""
    from aixeval.ingest.pipeline import ingest_any  # noqa: PLC0415
    from aixeval.scorecard.metrics import score_metrics  # noqa: PLC0415
    from aixeval.scorecard.profile import build_profile  # noqa: PLC0415

    result = ingest_any({"expected": "Paris", "output": "Paris"})
    scored = score_metrics(result.record)
    profile = build_profile(scored)
    bundle = build_run_bundle(result, scored, profile)
    print(f"bundle keys: {sorted(bundle.keys())}")
    print(f"schema_version: {bundle['schema_version']}")
    print(f"capability grade: {bundle['profile']['capability']}")

    import tempfile  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "run.json"
        write_run_bundle(bundle, path)
        read_back = read_run_bundle(path)
        print(f"round-tripped, same record: {read_back['record'] == bundle['record']}")

        replay = replay_bundle(read_back)
        print(f"replay matches, no model call: {replay['matches']}")


if __name__ == "__main__":
    print_bundle_example()
