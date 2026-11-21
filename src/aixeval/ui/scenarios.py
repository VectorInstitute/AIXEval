"""Folder-based demo scenarios: one folder per prefilled run, no Streamlit import.

Every directory under the scenarios root that holds a trace file becomes
one scenario (nested folders too, so ``scenarios/hiring/unconfirmed/``
works). Inside a scenario folder, only the trace is required::

    trace.json | trace.eval  the run (or the folder's only other .json/.eval)
    declared.json            declared fields (gold answers, safety flags, ...)
    overlay.json             multi-agent overlay (gold roles, faults, ...)
    scenario.json            title, domain, condition, story, description, metrics
    saved_run.json           a past run's record + judge rationale (+ any hard-stop
                             incident analysis), for replay
    image.png | image.jpg    reference image for cross_modal_hallucination

File names live in the constants below so the convention can change in
one place. A folder that can't be loaded is reported back as a problem
string rather than raised, so one broken scenario never hides the rest.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aixeval.ingest.pipeline import IngestResult, ingest_any, ingest_inspect_eval


SCENARIOS_DIR_ENV = "AIXEVAL_SCENARIOS_DIR"
TRACE_STEM = "trace"
TRACE_SUFFIXES = (".json", ".eval")
DECLARED_FILE = "declared.json"
OVERLAY_FILE = "overlay.json"
METADATA_FILE = "scenario.json"
SAVED_RUN_FILE = "saved_run.json"
SAVED_ANALYSIS_KEY = "hard_stop_analysis"
IMAGE_STEM = "image"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
_SIDECAR_FILES = frozenset({DECLARED_FILE, OVERLAY_FILE, METADATA_FILE, SAVED_RUN_FILE})
_NON_TRACE_JSON = _SIDECAR_FILES | frozenset(
    {
        "manifest.json",
        "coverage_report.json",
        "coverage_report.oracle.json",
    }
)

# A scenario's ``condition`` is a workflow type, optionally followed by
# comma-separated qualifiers: "Confirm gate, two runs".
WORKFLOW_DEFINITIONS: dict[str, str] = {
    "Sequential": "Agents or steps run one after another; each hands its result to the next.",
    "Parallel then join": "Several agents work on separate pieces at the same time, then one agent combines "
    "their results into the answer.",
    "Confirm gate": "The task includes a high-impact action (sending, ordering, filing) that must wait for a "
    "human's approval before it happens.",
    "Loop": "An agent repeats a step until a condition is met. Tests that it makes progress and stops, rather "
    "than retrying the same thing forever.",
    "Injection": "Something the agent reads (a document, web page or tool result) hides an instruction. Tests "
    "that the agent treats it as data, not as a command.",
}
QUALIFIER_DEFINITIONS: dict[str, str] = {
    "two runs": "Shipped as a pair, one run handled well and one badly, so the Compare page can show what "
    "changes on the card.",
    "single-agent baseline": "The same task done by one agent, to show what the multi-agent version adds.",
}


@dataclass(frozen=True)
class Scenario:
    """One prefilled demo run, loaded from its folder."""

    id: str
    folder: Path
    title: str
    domain: str
    condition: str
    story: str
    description: str
    trace_path: Path
    declared_fields: dict[str, Any] | None
    overlay: dict[str, Any] | None
    image_path: Path | None
    metric_ids: tuple[str, ...] | None
    saved_record: dict[str, Any] | None
    saved_rationale: dict[str, str]
    saved_analysis: dict[str, Any] | None = None

    @property
    def has_saved_run(self) -> bool:
        """Whether this scenario can be replayed without any judge call."""
        return self.saved_record is not None


def condition_definition(condition: str) -> str | None:
    """Explain a condition label, or return ``None`` if no part is known."""
    workflow, *qualifiers = (part.strip() for part in condition.split(","))
    sentences = [WORKFLOW_DEFINITIONS[workflow]] if workflow in WORKFLOW_DEFINITIONS else []
    sentences += [QUALIFIER_DEFINITIONS[q] for q in qualifiers if q in QUALIFIER_DEFINITIONS]
    return " ".join(sentences) or None


def default_scenarios_dir() -> Path:
    """Return ``$AIXEVAL_SCENARIOS_DIR`` if set, else ``fixtures/scenarios``."""
    override = os.environ.get(SCENARIOS_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[3] / "fixtures" / "scenarios"


def discover_scenarios(root: str | Path) -> tuple[list[Scenario], list[str]]:
    """Load every scenario folder under ``root``.

    Returns ``(scenarios, problems)``: scenarios sorted by domain then
    title, and one human-readable message per folder that looked like a
    scenario but failed to load. A missing ``root`` is a problem, not
    an exception.
    """
    root_path = Path(root).expanduser()
    if not root_path.is_dir():
        return [], [f"Scenarios folder not found: {root_path}"]

    scenarios: list[Scenario] = []
    problems: list[str] = []
    for folder in sorted({root_path, *(p for p in root_path.rglob("*") if p.is_dir())}):
        try:
            if _find_trace(folder) is None:
                continue
            scenarios.append(load_scenario(folder, root=root_path))
        except (ValueError, OSError) as exc:
            problems.append(f"{folder.relative_to(root_path)}: {exc}")
    scenarios.sort(key=lambda s: (s.domain.lower(), s.title.lower()))
    return scenarios, problems


def load_scenario(folder: str | Path, *, root: str | Path | None = None) -> Scenario:
    """Load one scenario folder.

    ``root``, when given, is used to derive the scenario's ``id`` (its
    path relative to ``root``) and a default ``domain`` (its top-level
    folder under ``root``, for nested layouts).

    Raises
    ------
    ValueError
        No trace file, more than one candidate trace file, or a sidecar
        file that isn't the JSON shape it should be.
    """
    folder_path = Path(folder)
    trace_path = _find_trace(folder_path)
    if trace_path is None:
        msg = f"no trace file ({TRACE_STEM}.json or {TRACE_STEM}.eval) in {folder_path}"
        raise ValueError(msg)

    relative = folder_path.relative_to(root) if root is not None else Path(folder_path.name)
    scenario_id = relative.as_posix() if relative.parts else folder_path.name
    metadata = _read_json_object(folder_path / METADATA_FILE) or {}
    saved_run = _read_json_object(folder_path / SAVED_RUN_FILE)
    metric_ids = _metric_ids(metadata, saved_run)

    default_domain = relative.parts[0] if len(relative.parts) > 1 else "other"
    return Scenario(
        id=scenario_id,
        folder=folder_path,
        title=str(metadata.get("title") or folder_path.name.replace("_", " ").replace("-", " ").capitalize()),
        domain=str(metadata.get("domain") or default_domain),
        condition=str(metadata.get("condition") or ""),
        story=str(metadata.get("story") or ""),
        description=str(metadata.get("description") or ""),
        trace_path=trace_path,
        declared_fields=_read_json_object(folder_path / DECLARED_FILE),
        overlay=_read_json_object(folder_path / OVERLAY_FILE),
        image_path=_find_image(folder_path),
        metric_ids=metric_ids,
        saved_record=_saved_record(saved_run),
        saved_rationale=_saved_rationale(saved_run),
        saved_analysis=_saved_analysis(saved_run),
    )


def ingest_scenario(scenario: Scenario) -> list[IngestResult]:
    """Ingest a scenario's trace with its declared fields and overlay applied.

    Returns one result per run, the same list shape the upload path
    produces (several for a multi-epoch ``.eval`` file, one otherwise).
    """
    if scenario.trace_path.suffix == ".eval":
        return ingest_inspect_eval(scenario.trace_path.read_bytes(), declared_fields=scenario.declared_fields)
    payload = json.loads(scenario.trace_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"{scenario.trace_path.name} must hold a JSON object"
        raise ValueError(msg)
    return [ingest_any(payload, declared_fields=scenario.declared_fields, overlay=scenario.overlay)]


def read_trace_payload(scenario: Scenario) -> dict[str, Any] | None:
    """Return the scenario's raw trace dict, or ``None`` for an ``.eval`` archive."""
    if scenario.trace_path.suffix == ".eval":
        return None
    payload = json.loads(scenario.trace_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _find_trace(folder: Path) -> Path | None:
    for suffix in TRACE_SUFFIXES:
        named = folder / f"{TRACE_STEM}{suffix}"
        if named.is_file():
            return named
    candidates = [
        p for p in folder.iterdir() if p.is_file() and p.suffix in TRACE_SUFFIXES and p.name not in _NON_TRACE_JSON
    ]
    if len(candidates) > 1:
        names = ", ".join(sorted(p.name for p in candidates))
        msg = f"several candidate trace files ({names}); name the trace {TRACE_STEM}.json"
        raise ValueError(msg)
    return candidates[0] if candidates else None


def _find_image(folder: Path) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        named = folder / f"{IMAGE_STEM}{suffix}"
        if named.is_file():
            return named
    return None


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"{path.name} is not valid JSON ({exc})"
        raise ValueError(msg) from exc
    if not isinstance(data, dict):
        msg = f"{path.name} must hold a JSON object"
        raise ValueError(msg)
    return data


def _metric_ids(metadata: dict[str, Any], saved_run: dict[str, Any] | None) -> tuple[str, ...] | None:
    """Pick metrics to pre-select: the listed ones, else what the saved run scored."""
    listed = metadata.get("metrics")
    if listed is not None:
        if not isinstance(listed, list) or not all(isinstance(m, str) for m in listed):
            msg = f"{METADATA_FILE} 'metrics' must be a list of metric ids"
            raise ValueError(msg)
        return tuple(listed)
    if saved_run is None:
        return None
    rows = saved_run.get("rows")
    row_list = list(rows.values()) if isinstance(rows, dict) else rows
    if not isinstance(row_list, list):
        return None
    return tuple(row["metric_id"] for row in row_list if isinstance(row, dict) and row.get("status") == "scored")


def _saved_record(saved_run: dict[str, Any] | None) -> dict[str, Any] | None:
    if saved_run is None:
        return None
    record = saved_run.get("record")
    if not isinstance(record, dict):
        msg = f"{SAVED_RUN_FILE} must have a 'record' object"
        raise ValueError(msg)
    return record


def _saved_rationale(saved_run: dict[str, Any] | None) -> dict[str, str]:
    if saved_run is None:
        return {}
    rationale = saved_run.get("rationale") or {}
    if not isinstance(rationale, dict):
        msg = f"{SAVED_RUN_FILE} 'rationale' must be an object"
        raise ValueError(msg)
    return {str(k): str(v) for k, v in rationale.items()}


def _saved_analysis(saved_run: dict[str, Any] | None) -> dict[str, Any] | None:
    if saved_run is None:
        return None
    analysis = saved_run.get(SAVED_ANALYSIS_KEY)
    if analysis is not None and not isinstance(analysis, dict):
        msg = f"{SAVED_RUN_FILE} '{SAVED_ANALYSIS_KEY}' must be an object"
        raise ValueError(msg)
    return analysis
