"""A thin command-line entrypoint for scoring, reporting, and comparing saved runs.

Salvaged from the old ``feature/portable-eval-kernel`` vision's CLI idea
(``run``/``validate``/``reband``/``rescore``/``rerun``), kept narrow on
purpose, same as ``bundle.py`` and ``reband.py``: commands wrapping
the library calls that already exist (``ingest_any``/``score_metrics``/
``build_profile``, plus ``build_run_bundle``/``reband_profile``/
``replay_bundle``/``compare_bundles``/``score_cross_run_consistency``),
not a new scheduler or a new execution model. No new dependency: stdlib
``argparse``.

    aixeval score  <dump.json> [--declared-fields FIELDS.json] \\
        [--overlay OVERLAY.json] [--metrics id,id] \\
        [--live-judges grounding,refusal] [--audit-contamination] \\
        [--test-retest-metric grounding --test-retest-n 3] [--out run.json]
    aixeval report <run.json>
    aixeval reband <run.json> [--cutoffs 0.2,0.4,0.6,0.8] \\
        [--hard-stop unsafe_action_cap=0.15]
    aixeval replay <run.json>
    aixeval compare <run_a.json> <run_b.json> [--max-grade-drift 1]
    aixeval cross-run-consistency <suite_dir> --metric grounding \\
        [--bootstrap-samples 2000] [--seed 0]

``compare`` is software support for an independent-rerun safeguard:
it diffs two already-produced bundles, it does not re-execute
anything. See
``aixeval.bundle.compare_bundles`` for exactly what it proves
and what it doesn't.

``cross-run-consistency`` is the same non-reexecuting philosophy one
level up: ``<suite_dir>`` holds one subdirectory per problem, each
holding two or more already-produced run bundles of that same problem
(e.g. from running ``aixeval score --live-judges ... --out`` several
times). This command reads them, groups them by subdirectory name, and
reports one variance band for the whole set -- it does not call a
judge or re-execute anything either. See
``aixeval.cross_run_consistency`` for what the band means and how
"unstable" is decided.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from aixeval.bundle import build_run_bundle, compare_bundles, read_run_bundle, replay_bundle, write_run_bundle
from aixeval.cross_run_consistency import ProblemRuns, score_cross_run_consistency
from aixeval.ingest.pipeline import IngestResult, ingest_any, ingest_inspect_eval
from aixeval.reband import reband_profile, reband_results
from aixeval.scorecard.hard_stops import explain_hard_stops
from aixeval.scorecard.metrics import ALL_METRIC_IDS, score_metrics
from aixeval.scorecard.profile import (
    DIMENSIONS,
    HARD_STOP_RULES,
    HIGHER_IS_BETTER_CUTOFFS,
    HardStopRule,
    build_profile,
    label_for_band,
)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the selected subcommand.

    Returns the process exit code: 0 on success, 1 on a handled error
    (bad input, missing file). Never lets a user-facing input mistake
    surface as a raw Python traceback.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aixeval", description="Score, report on, reband, and replay saved agent runs."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    score_p = sub.add_parser("score", help="Ingest and score a dump or trace file.")
    score_p.add_argument("dump", type=Path, help="A JSON dump/trace, or an Inspect AI .eval file.")
    score_p.add_argument(
        "--declared-fields",
        type=Path,
        default=None,
        help="JSON file of declared fields. If omitted, uses declared.json next to the dump when that file exists.",
    )
    score_p.add_argument(
        "--overlay",
        type=Path,
        default=None,
        help=(
            "Multi-agent overlay JSON (gold roles, faults, MAS step counts). "
            "If omitted, uses overlay.json next to the dump when that file exists."
        ),
    )
    score_p.add_argument(
        "--metrics",
        default=None,
        help=(
            "Comma-separated metric ids to score. If omitted, uses scenario.json's "
            "metrics list next to the dump when present; otherwise the full catalog."
        ),
    )
    score_p.add_argument(
        "--live-judges",
        default=None,
        help="Comma-separated live-judge metric ids to run (needs a configured judge provider; see judge/config.py).",
    )
    score_p.add_argument(
        "--audit-contamination",
        action="store_true",
        help=(
            "Check whether gold/expected text already appears in a trace's inputs "
            "(same as the UI contamination-audit checkbox). Off by default."
        ),
    )
    score_p.add_argument(
        "--test-retest-metric",
        default=None,
        metavar="METRIC_ID",
        help=(
            "Call one live-judge metric repeatedly on the same input and score "
            "test_retest_stability from the spread (e.g. grounding). Needs a configured judge provider."
        ),
    )
    score_p.add_argument(
        "--test-retest-n", type=int, default=3, help="How many times to call --test-retest-metric. Default 3."
    )
    score_p.add_argument("--out", type=Path, default=None, help="Write a run bundle (see bundle.py) to this path.")

    report_p = sub.add_parser("report", help="Print a human-readable report from a saved run bundle.")
    report_p.add_argument("bundle", type=Path)

    reband_p = sub.add_parser(
        "reband", help="Re-grade a saved run bundle under a different policy: no rescoring, no model calls."
    )
    reband_p.add_argument("bundle", type=Path)
    reband_p.add_argument("--cutoffs", default=None, help="Four comma-separated grade cutoffs, e.g. 0.2,0.4,0.6,0.8")
    reband_p.add_argument(
        "--hard-stop",
        action="append",
        default=[],
        metavar="NAME=CUTOFF",
        help="Override one hard-stop rule's cutoff, e.g. unsafe_action_cap=0.15. Repeatable.",
    )

    replay_p = sub.add_parser(
        "replay", help="Re-score a saved bundle's own record and confirm it reproduces the saved card: no model calls."
    )
    replay_p.add_argument("bundle", type=Path)

    compare_p = sub.add_parser(
        "compare",
        help=(
            "Diff two independently-produced bundles of the same task: software support for an "
            "independent-rerun safeguard. Does not re-execute anything itself."
        ),
    )
    compare_p.add_argument("bundle_a", type=Path)
    compare_p.add_argument("bundle_b", type=Path)
    compare_p.add_argument("--max-grade-drift", type=int, default=1, help="Grade bands tolerated before flagging.")

    consistency_p = sub.add_parser(
        "cross-run-consistency",
        help=(
            "Band the run-to-run variance of one metric across many problems' repeated bundles. "
            "Does not re-execute anything itself."
        ),
    )
    consistency_p.add_argument(
        "dir",
        type=Path,
        help="A directory with one subdirectory per problem, each holding >=2 run-bundle .json files.",
    )
    consistency_p.add_argument(
        "--metric", required=True, metavar="METRIC_ID", help="Which metric's raw_score to read from each bundle."
    )
    consistency_p.add_argument(
        "--bootstrap-samples", type=int, default=2000, help="Bootstrap resamples for the confidence interval."
    )
    consistency_p.add_argument("--seed", type=int, default=None, help="Seed the bootstrap for a reproducible interval.")
    return parser


def _cmd_score(args: argparse.Namespace) -> int:
    declared_fields = (
        _json_object(args.declared_fields) if args.declared_fields else _sidecar_json(args.dump, "declared.json")
    )
    overlay = _json_object(args.overlay) if args.overlay else _sidecar_json(args.dump, "overlay.json")
    selected_ids = _selected_metric_ids(args)
    live_judge_ids = {m.strip() for m in args.live_judges.split(",") if m.strip()} if args.live_judges else set()
    if selected_ids is not None:
        selected_ids = set(selected_ids) | live_judge_ids
        if args.test_retest_metric:
            selected_ids.add(args.test_retest_metric)
            selected_ids.add("test_retest_stability")

    payload: dict[str, Any] | None
    if args.dump.suffix == ".eval":
        results = ingest_inspect_eval(args.dump.read_bytes(), declared_fields=declared_fields)
        payload = None
    else:
        payload = json.loads(args.dump.read_text(encoding="utf-8"))
        results = [
            ingest_any(
                payload,
                declared_fields=declared_fields,
                overlay=overlay,
                audit_contamination=args.audit_contamination,
            )
        ]

    for i, result in enumerate(results):
        if len(results) > 1:
            print(f"=== epoch {i + 1} of {len(results)} ===")
        if args.audit_contamination and args.dump.suffix != ".eval":
            if result.contamination_evidence:
                print(f"contamination flagged: {result.contamination_evidence}")
            elif result.spans is None:
                print("contamination audit: skipped (not a trace)")
            else:
                print("contamination audit: no leak found")
        record = _apply_live_judges(result, live_judge_ids, payload)
        record = _apply_test_retest(record, result, args.test_retest_metric, args.test_retest_n, payload)
        scored = score_metrics(record)
        if selected_ids is not None:
            scored = [row for row in scored if row["metric_id"] in selected_ids]
        profile_rebanded = reband_profile(scored)  # shipped policy, just reuses build_profile under the hood
        models = _judge_models(record)
        _print_card(profile_rebanded, scored_results=scored, judge=models)
        _print_hard_stop_explanation(scored, record)
        if args.out:
            from dataclasses import replace  # noqa: PLC0415

            out_path = args.out if len(results) == 1 else args.out.with_stem(f"{args.out.stem}_epoch{i + 1}")
            bundle = build_run_bundle(
                replace(result, record=record),
                scored,
                profile_rebanded,
                judge=", ".join(models) if models else None,
            )
            write_run_bundle(bundle, out_path)
            print(f"wrote {out_path}")
        print()
    return 0


def _apply_live_judges(
    result: IngestResult, live_judge_ids: set[str], payload: dict[str, Any] | None
) -> dict[str, Any]:
    record = dict(result.record)
    if not live_judge_ids:
        return record
    from aixeval.judge.config import build_client_from_env  # noqa: PLC0415
    from aixeval.judge.live import extract_live_grounding_evidence, run_live_judges  # noqa: PLC0415

    client = build_client_from_env()
    record, _rationale, errors = run_live_judges(
        record,
        trajectory=result.trajectory,
        image_path=None,
        selected_ids=live_judge_ids,
        client=client,
        evidence=extract_live_grounding_evidence(payload, result.spans),
    )
    for metric_id, message in errors.items():
        print(f"warning: {metric_id} live judge call failed: {message}", file=sys.stderr)
    return record


def _apply_test_retest(
    record: dict[str, Any],
    result: IngestResult,
    metric_id: str | None,
    n: int,
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """Call one live-judge metric ``n`` times, attaching the scores as ``rerun_scores``.

    The missing glue between ``judge.runner.collect_rerun_scores`` and
    this CLI: adapts ``judge.live.run_live_judges`` (one call per
    invocation, trajectory/image-path already wired, same dispatch
    ``--live-judges`` uses) into the zero-arg ``() -> (record,
    attachment)`` shape ``collect_rerun_scores`` expects, by
    synthesizing a minimal ``{"status": ...}`` attachment from
    ``run_live_judges``'s own error dict.
    """
    if metric_id is None:
        return record
    from aixeval.judge.config import build_client_from_env  # noqa: PLC0415
    from aixeval.judge.live import LIVE_JUDGE_METRICS, extract_live_grounding_evidence, run_live_judges  # noqa: PLC0415
    from aixeval.judge.runner import collect_rerun_scores  # noqa: PLC0415

    spec = LIVE_JUDGE_METRICS.get(metric_id)
    if spec is None:
        msg = f"Unknown live-judge metric {metric_id!r} for --test-retest-metric. Known: {sorted(LIVE_JUDGE_METRICS)}"
        raise ValueError(msg)
    client = build_client_from_env()

    # Each call still reads the live judge's actual verdict; these are
    # real, paid calls, so the last successful one's verdict is kept on
    # the record too (not just the derived variance), the same way a
    # single --live-judges call would have left it.
    last_scored: dict[str, Any] = record

    def run_once() -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal last_scored
        updated, _rationale, errors = run_live_judges(
            record,
            trajectory=result.trajectory,
            image_path=None,
            selected_ids={metric_id},
            client=client,
            evidence=extract_live_grounding_evidence(payload, result.spans),
        )
        if metric_id in errors:
            return updated, {"status": "failed"}
        last_scored = updated
        return updated, {"status": "scored"}

    scores = collect_rerun_scores(run_once, spec.record_field, n=n)
    if len(scores) < 2:
        print(
            f"warning: only {len(scores)} successful {metric_id} rerun(s): test_retest_stability will be N/A",
            file=sys.stderr,
        )
        return last_scored
    print(f"test-retest: {metric_id} scored {scores} across {n} calls")
    return {**last_scored, "rerun_scores": scores}


def _cmd_report(args: argparse.Namespace) -> int:
    bundle = read_run_bundle(args.bundle)
    print(f"source: {bundle['source_kind']}")
    if bundle.get("judge"):
        print(f"judge: {bundle['judge']}")
    if bundle.get("contamination_evidence"):
        print(f"contamination flagged: {bundle['contamination_evidence']}")
    _print_card(bundle["profile"], scored_results=bundle["scored_results"], judge=None)
    _print_hard_stop_explanation(bundle["scored_results"], bundle["record"])
    _print_metric_rows(bundle["scored_results"])
    return 0


def _cmd_reband(args: argparse.Namespace) -> int:
    bundle = read_run_bundle(args.bundle)
    cutoffs = tuple(float(c) for c in args.cutoffs.split(",")) if args.cutoffs else None
    hard_stop_rules = dict(HARD_STOP_RULES)
    for override in args.hard_stop:
        name, _, cutoff_text = override.partition("=")
        if name not in HARD_STOP_RULES:
            msg = f"Unknown hard-stop rule {name!r}. Known: {sorted(HARD_STOP_RULES)}"
            raise ValueError(msg)
        hard_stop_rules[name] = {**HARD_STOP_RULES[name], "cutoff": float(cutoff_text)}

    kwargs: dict[str, Any] = {"hard_stop_rules": hard_stop_rules}
    if cutoffs is not None:
        kwargs["cutoffs"] = cutoffs
    rows = reband_results(
        bundle["scored_results"],
        cutoffs=cutoffs if cutoffs is not None else HIGHER_IS_BETTER_CUTOFFS,
    )
    rebanded = reband_profile(bundle["scored_results"], **kwargs)
    _print_card(rebanded, scored_results=rows)
    _print_metric_rows(rows)
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    bundle = read_run_bundle(args.bundle)
    result = replay_bundle(bundle)
    if result["matches"]:
        print("replay matches exactly: no model calls, same record in, same card out.")
        return 0
    print("replay MISMATCH:")
    for diff in result["metric_diffs"]:
        print(f"  metric {diff['metric_id']}: saved={diff['saved']} replayed={diff['replayed']}")
    for diff in result["profile_diffs"]:
        print(f"  dimension {diff['dimension']}: saved={diff['saved']} replayed={diff['replayed']}")
    return 1


def _cmd_compare(args: argparse.Namespace) -> int:
    bundle_a = read_run_bundle(args.bundle_a)
    bundle_b = read_run_bundle(args.bundle_b)
    result = compare_bundles(bundle_a, bundle_b, max_grade_drift=args.max_grade_drift)
    if not result["diverged"]:
        print(f"no dimension drifted by more than {args.max_grade_drift} grade band(s): re-run holds up.")
        return 0
    print(f"DIVERGED beyond {args.max_grade_drift} grade band(s):")
    for diff in result["dimension_diffs"]:
        if diff["drift"] is None:
            print(f"  {diff['dimension']}: {diff['grade_a']} vs {diff['grade_b']} (N/A vs scored)")
        else:
            print(f"  {diff['dimension']}: {diff['grade_a']} vs {diff['grade_b']} (drift {diff['drift']})")
    return 1


def _cmd_cross_run_consistency(args: argparse.Namespace) -> int:
    problems = _collect_problem_runs(args.dir)
    result = score_cross_run_consistency(
        problems, metric_id=args.metric, bootstrap_samples=args.bootstrap_samples, seed=args.seed
    )
    if result["status"] != "scored":
        print(f"could not score: {result.get('na_reason', 'unknown')}")
        _print_excluded(result.get("excluded_problems", []))
        return 1

    print(f"{args.metric}: grade {result['grade']}/4 ({label_for_band(result['grade'])})")
    print(f"  mean_variance:   {result['mean_variance']:.4f} (raw_score {result['raw_score']:.4f})")
    print(f"  problems_scored: {result['problems_scored']}")
    print(f"  unstable_share:  {result['unstable_share']:.0%}")
    if result["bootstrap_ci"] is not None:
        low, high = result["bootstrap_ci"]
        print(f"  bootstrap_ci:    ({low:.4f}, {high:.4f})")
    if result["unstable_problems"]:
        names = ", ".join(p["problem_id"] for p in result["unstable_problems"])
        print(f"  not fully stable: {names}")
    _print_excluded(result.get("excluded_problems", []))
    return 0


def _print_excluded(excluded: list[dict[str, Any]]) -> None:
    if not excluded:
        return
    print("  excluded (too few usable runs):")
    for entry in excluded:
        print(f"    {entry['problem_id']}: {entry['runs_available']} usable run(s)")


def _collect_problem_runs(suite_dir: Path) -> list[ProblemRuns]:
    """One ``ProblemRuns`` per immediate subdirectory of ``suite_dir``.

    Each subdirectory's name is the problem id; every ``*.json`` file
    directly inside it is read as a run bundle (``read_run_bundle``,
    so a malformed or wrong-schema file raises immediately, same as
    every other command here). A subdirectory with zero bundle files
    is skipped outright, not passed through as an empty problem:
    ``score_cross_run_consistency`` only needs to decide what to do
    with *too few* runs, not none at all.
    """
    if not suite_dir.is_dir():
        msg = f"{suite_dir} is not a directory"
        raise ValueError(msg)
    problems: list[ProblemRuns] = []
    for subdir in sorted(p for p in suite_dir.iterdir() if p.is_dir()):
        bundle_paths = sorted(subdir.glob("*.json"))
        if not bundle_paths:
            continue
        bundles = [read_run_bundle(path) for path in bundle_paths]
        problems.append(ProblemRuns(subdir.name, bundles))
    return problems


def _json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = f"{path} must hold a JSON object"
        raise ValueError(msg)
    return data


def _sidecar_json(dump: Path, name: str) -> dict[str, Any] | None:
    path = dump.parent / name
    if not path.is_file():
        return None
    return _json_object(path)


def _parse_metric_ids(raw: str, *, source: str) -> set[str]:
    ids = {item.strip() for item in raw.split(",") if item.strip()}
    unknown = sorted(ids - set(ALL_METRIC_IDS))
    if unknown:
        msg = f"Unknown metric id(s) in {source}: {unknown}"
        raise ValueError(msg)
    return ids


def _selected_metric_ids(args: argparse.Namespace) -> set[str] | None:
    if args.metrics:
        return _parse_metric_ids(args.metrics, source="--metrics")
    scenario_path = args.dump.parent / "scenario.json"
    if not scenario_path.is_file():
        return None
    listed = _json_object(scenario_path).get("metrics")
    if listed is None:
        return None
    if not isinstance(listed, list) or not all(isinstance(item, str) for item in listed):
        msg = f"{scenario_path.name} 'metrics' must be a list of metric ids"
        raise ValueError(msg)
    unknown = sorted(set(listed) - set(ALL_METRIC_IDS))
    if unknown:
        msg = f"Unknown metric id(s) in {scenario_path.name}: {unknown}"
        raise ValueError(msg)
    return set(listed)


def _disabled_hard_stop_rules() -> dict[str, HardStopRule]:
    """Shipped rules with cutoffs that cannot fire, used to print an uncapped card."""
    disabled: dict[str, HardStopRule] = {}
    for name, rule in HARD_STOP_RULES.items():
        updated: HardStopRule = {**rule, "cutoff": 2.0 if rule["compare"] == "ge" else -1.0}
        disabled[name] = updated
    return disabled


def _print_metric_rows(rows: list[dict[str, Any]]) -> None:
    print()
    print("per-metric detail:")
    for row in sorted(rows, key=lambda item: item["metric_id"]):
        if row["status"] == "scored":
            print(
                f"  {row['metric_id']}: {row['raw_score']:.3f} -> grade {row['grade']}/4 ({label_for_band(row['grade'])})"
            )
        else:
            print(f"  {row['metric_id']}: N/A ({row.get('na_reason', 'unknown')})")


def _print_card(
    profile: dict[str, Any],
    *,
    scored_results: list[dict[str, Any]] | None = None,
    judge: tuple[str, ...] | None = None,
) -> None:
    for dimension in DIMENSIONS:
        grade = profile[dimension]
        print(f"  {dimension}: {'N/A' if grade is None else f'{grade}/4 ({label_for_band(grade)})'}")
    flags = tuple(profile.get("applied_hard_stops") or ())
    print(f"  hard stops: {', '.join(flags) if flags else '(none)'}")
    if judge:
        print(f"  judge: {', '.join(judge)}")
    if flags and scored_results is not None:
        uncapped = build_profile(scored_results, hard_stop_rules=_disabled_hard_stop_rules())
        bits = []
        for dimension in DIMENSIONS:
            capped, before = profile[dimension], uncapped[dimension]
            if capped == before:
                continue
            before_text = "N/A" if before is None else f"{before}/4"
            bits.append(f"{dimension} {before_text}")
        if bits:
            print(f"  uncapped (before hard-stop caps): {', '.join(bits)}")
    if all(profile[dimension] is None for dimension in DIMENSIONS):
        print(
            "  note: every pillar is N/A because no selected metric scored. "
            "Pass --declared-fields / --overlay, or put declared.json and overlay.json next to the dump."
        )


def _print_hard_stop_explanation(rows: list[dict[str, Any]], record: dict[str, Any]) -> None:
    explanation = explain_hard_stops(rows, record)
    for stop in explanation["hard_stops"]:
        print(f"  {stop['hard_stop']}: {stop['definition']}")
        print(f"    {stop['rule']}; {stop['effect']}")
        evidence = stop.get("evidence_fields") or {}
        if evidence:
            bits = [f"{name}={value!r}" for name, value in evidence.items()]
            print(f"    evidence: {', '.join(bits)}")


def _judge_models(record: dict[str, Any]) -> tuple[str, ...]:
    from aixeval.judge.provenance import models_from_provenance  # noqa: PLC0415

    return models_from_provenance(record)


# One lookup, not an if-chain: keeps main() a single early-return dispatch
# no matter how many subcommands this CLI grows.
_COMMANDS: dict[str, Any] = {
    "score": _cmd_score,
    "report": _cmd_report,
    "reband": _cmd_reband,
    "replay": _cmd_replay,
    "compare": _cmd_compare,
    "cross-run-consistency": _cmd_cross_run_consistency,
}


if __name__ == "__main__":
    sys.exit(main())
