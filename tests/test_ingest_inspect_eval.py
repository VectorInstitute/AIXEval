"""ingest_inspect_eval: one IngestResult per sample/epoch in an Inspect .eval file."""

import io
import json
import zipfile
from pathlib import Path

import pytest

from aixeval.ingest.pipeline import ingest_inspect_eval


FIXTURES = Path(__file__).parent.parent / "fixtures"


def _sample(epoch: int, **overrides: object) -> dict:
    base = json.loads((FIXTURES / "inspect_sample.json").read_text(encoding="utf-8"))
    base["epoch"] = epoch
    base.update(overrides)
    return base


def _eval_bytes(*samples: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("header.json", b"{}")
        for sample in samples:
            archive.writestr(f"samples/{sample['id']}_epoch_{sample['epoch']}.json", json.dumps(sample))
    return buf.getvalue()


def test_one_result_per_epoch() -> None:
    data = _eval_bytes(_sample(1), _sample(2), _sample(3))
    results = ingest_inspect_eval(data)
    assert len(results) == 3
    for result in results:
        assert result.source_kind == "trace"
        assert result.spans is not None
        assert result.trajectory is not None


def test_output_and_steps_are_auto_derived_from_the_message_trace() -> None:
    data = _eval_bytes(_sample(1))
    [result] = ingest_inspect_eval(data)
    assert result.record["output"] == "Fixed the off-by-one bug in add(); test_add now passes."
    assert result.record["steps"] == len(result.spans)


def test_non_empty_target_becomes_expected() -> None:
    data = _eval_bytes(_sample(1, target="tests pass"))
    [result] = ingest_inspect_eval(data)
    assert result.record["expected"] == "tests pass"


def test_empty_target_is_not_fabricated_into_expected() -> None:
    data = _eval_bytes(_sample(1, target=""))
    [result] = ingest_inspect_eval(data)
    assert "expected" not in result.record


def test_declared_fields_apply_to_every_epoch() -> None:
    data = _eval_bytes(_sample(1), _sample(2))
    results = ingest_inspect_eval(data, declared_fields={"should_refuse": False})
    assert all(result.record["should_refuse"] is False for result in results)


def test_unknown_declared_field_raises() -> None:
    data = _eval_bytes(_sample(1))
    with pytest.raises(ValueError, match="nonsense_field_name"):
        ingest_inspect_eval(data, declared_fields={"nonsense_field_name": True})
