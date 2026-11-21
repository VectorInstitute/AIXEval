"""Inspect AI .eval -> normalized spans, tested on a synthetic committed fixture."""

import io
import json
import zipfile
from pathlib import Path

import pytest

from aixeval.ingest.trace_inspect import parse_inspect_sample, read_inspect_eval


_FIXTURES = Path(__file__).parent.parent / "fixtures"


def _sample() -> dict:
    return json.loads((_FIXTURES / "inspect_sample.json").read_text(encoding="utf-8"))


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    """Build an in-memory zip (plain DEFLATE: no real Zstd needed for these tests)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buf.getvalue()


def test_assistant_and_tool_messages_become_llm_and_tool_spans() -> None:
    spans = {s.id: s for s in parse_inspect_sample(_sample())}
    assert spans["msg2"].kind == "llm"
    assert spans["msg3"].kind == "tool"
    # system/user messages (indices 0, 1) are not spans at all.
    assert "msg0" not in spans
    assert "msg1" not in spans


def test_tool_span_input_is_the_triggering_call_function_and_arguments() -> None:
    spans = {s.id: s for s in parse_inspect_sample(_sample())}
    assert spans["msg3"].input == "bash({'command': 'pytest -q'})"
    assert spans["msg3"].output == "1 failed: test_add expected 5 got 4"


def test_content_block_list_keeps_text_and_drops_reasoning() -> None:
    spans = {s.id: s for s in parse_inspect_sample(_sample())}
    # msg4's assistant turn has only a reasoning block, empty text -> no output.
    assert spans["msg4"].output is None
    # msg6's final turn has real text alongside reasoning -> only the text survives.
    assert spans["msg6"].output == "Fixed the off-by-one bug in add(); test_add now passes."


def test_missing_messages_list_raises() -> None:
    with pytest.raises(ValueError, match="no messages list"):
        parse_inspect_sample({"id": 1})


def test_read_inspect_eval_returns_sorted_sample_dicts() -> None:
    data = _zip_bytes(
        {
            "header.json": b"{}",
            "samples/1_epoch_2.json": json.dumps({"id": 1, "epoch": 2, "messages": []}).encode(),
            "samples/1_epoch_1.json": json.dumps({"id": 1, "epoch": 1, "messages": []}).encode(),
        }
    )
    samples = read_inspect_eval(data)
    assert [s["epoch"] for s in samples] == [1, 2]


def test_read_inspect_eval_raises_without_sample_entries() -> None:
    data = _zip_bytes({"header.json": b"{}"})
    with pytest.raises(ValueError, match="no samples"):
        read_inspect_eval(data)


def test_read_inspect_eval_raises_on_a_non_zip_file() -> None:
    with pytest.raises(ValueError, match="not a valid .eval file"):
        read_inspect_eval(b"not a zip at all")


def _two_samples() -> bytes:
    return _zip_bytes(
        {
            "samples/1_epoch_1.json": json.dumps({"id": 1, "epoch": 1, "messages": []}).encode(),
            "samples/2_epoch_1.json": json.dumps({"id": 2, "epoch": 1, "messages": []}).encode(),
        }
    )


def test_read_inspect_eval_rejects_an_oversized_sample_entry() -> None:
    with pytest.raises(ValueError, match="decompresses to more than the limit of 10 bytes"):
        read_inspect_eval(_two_samples(), max_sample_bytes=10)


def test_read_inspect_eval_rejects_samples_over_the_total_cap() -> None:
    one_sample = len(json.dumps({"id": 1, "epoch": 1, "messages": []}))
    with pytest.raises(ValueError, match="samples decompress to more than the limit"):
        read_inspect_eval(_two_samples(), max_total_bytes=one_sample + 1)


def test_read_inspect_eval_rejects_too_many_sample_entries() -> None:
    with pytest.raises(ValueError, match="more than the limit of 1"):
        read_inspect_eval(_two_samples(), max_entries=1)


def test_read_inspect_eval_cap_holds_against_a_highly_compressed_entry() -> None:
    # A small DEFLATE archive that expands far past the cap: the read stops
    # at the cap instead of inflating the whole member.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("samples/1_epoch_1.json", b" " * 5_000_000)
    assert len(buf.getvalue()) < 50_000
    with pytest.raises(ValueError, match="decompresses to more than the limit"):
        read_inspect_eval(buf.getvalue(), max_sample_bytes=1024)
