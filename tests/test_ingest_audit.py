"""audit_expected_in_inputs cancels only when enabled and gold is long enough."""

from pathlib import Path

from aixeval.ingest import (
    audit_expected_in_inputs,
    collect_langfuse_input_texts,
    ingest_langfuse,
)
from aixeval.ingest.load import load_json_object
from aixeval.scorecard import score_metrics


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_hit_cancels_and_marks_leak() -> None:
    record = {"expected": "Paris", "output": "Paris"}
    out, evidence = audit_expected_in_inputs(
        record,
        ("Capital of France? Paris is the answer.",),
        enabled=True,
    )
    assert out["cancelled"] is True
    assert out["contaminated"] is True
    assert out["evidence_status"] == "cancelled"
    assert evidence is not None
    assert "100%" in evidence
    assert "Paris is the answer" in evidence
    rows = score_metrics(out)
    by_id = {row["metric_id"]: row for row in rows}
    assert by_id["task_success"]["na_reason"] == "cancelled"
    assert by_id["verbalized_confidence"]["status"] == "na"


def test_gold_only_in_expected_does_not_cancel() -> None:
    record = {"expected": "Paris", "output": "Paris"}
    out, evidence = audit_expected_in_inputs(record, ("Capital of France?",), enabled=True)
    assert "cancelled" not in out
    assert evidence is None
    assert score_metrics(out)[0]["raw_score"] == 1.0


def test_off_by_default() -> None:
    record = {"expected": "Paris", "output": "Paris"}
    out, evidence = audit_expected_in_inputs(record, ("... Paris ...",))
    assert out == record
    assert evidence is None


def test_short_gold_is_skipped() -> None:
    record = {"expected": "yes", "output": "yes"}
    out, evidence = audit_expected_in_inputs(record, ("Say yes or no.",), enabled=True)
    assert "cancelled" not in out
    assert evidence is None


def test_no_expected_is_unchanged() -> None:
    record = {"output": "Paris"}
    out, evidence = audit_expected_in_inputs(record, ("Paris is here.",), enabled=True)
    assert out == record
    assert evidence is None


def test_langfuse_ingest_does_not_audit() -> None:
    record = ingest_langfuse(FIXTURES / "langfuse_gold_in_input.json")
    assert "cancelled" not in record
    assert record["expected"] == "Paris"
    assert record["output"] == "Paris"


def test_langfuse_helper_then_audit_cancels() -> None:
    path = FIXTURES / "langfuse_gold_in_input.json"
    record = ingest_langfuse(path)
    texts = collect_langfuse_input_texts(load_json_object(path))
    out, evidence = audit_expected_in_inputs(record, texts, enabled=True)
    assert out["cancelled"] is True
    assert any("Paris" in text for text in texts)
    assert evidence is not None


_LONG_EXPECTED = (
    "the clause is enforceable against sophisticated commercial parties because it was "
    "conspicuously disclosed in the body of the contract and not buried in fine print"
)


def test_partial_verbatim_leak_is_caught_even_though_full_string_never_appears() -> None:
    # Only the first 70% of a longer gold answer shows up, verbatim, inside
    # otherwise-unrelated text; the exact-substring check this replaced
    # would have missed this entirely, since the complete string is never
    # present anywhere in the input.
    words = _LONG_EXPECTED.split()
    leaked_part = " ".join(words[: int(len(words) * 0.7)])
    haystack = f"Context: {leaked_part} -- something totally different and unrelated here instead."
    out, evidence = audit_expected_in_inputs({"expected": _LONG_EXPECTED}, (haystack,), enabled=True)
    assert out["contaminated"] is True
    assert out["cancelled"] is True
    assert evidence is not None
    assert "%" in evidence
    assert leaked_part in evidence


def test_evidence_excerpt_is_truncated_for_a_very_long_input_text() -> None:
    huge_haystack = f"Context: {_LONG_EXPECTED} " + ("filler " * 200)
    _, evidence = audit_expected_in_inputs({"expected": _LONG_EXPECTED}, (huge_haystack,), enabled=True)
    assert evidence is not None
    assert "..." in evidence
    assert len(evidence) < len(huge_haystack)


def test_a_few_scattered_words_in_common_does_not_cancel() -> None:
    # A handful of words gold happens to share with an unrelated input text
    # must not look like a leak; the windows that matter are 8 words long,
    # not single words, so incidental overlap stays well under threshold.
    haystack = "The contract was enforceable and disclosed, but under entirely different facts and reasoning."
    out, evidence = audit_expected_in_inputs({"expected": _LONG_EXPECTED}, (haystack,), enabled=True)
    assert "cancelled" not in out
    assert evidence is None


def test_full_exact_leak_of_a_long_answer_still_cancels() -> None:
    # Backward-compatible with the prior exact-substring behavior for the
    # case it always handled: the whole gold text, verbatim, in the input.
    haystack = f"Context: {_LONG_EXPECTED} -- now answer the question."
    out, evidence = audit_expected_in_inputs({"expected": _LONG_EXPECTED}, (haystack,), enabled=True)
    assert out["contaminated"] is True
    assert evidence is not None
    assert "100%" in evidence


def test_overlap_is_checked_per_input_text_not_summed_across_them() -> None:
    # Two input texts that would each individually fall short of the
    # threshold must not combine into a hit just because gold's windows
    # are spread across both of them.
    words = _LONG_EXPECTED.split()
    midpoint = len(words) // 2
    first_half_text = "Unrelated: " + " ".join(words[:midpoint])
    second_half_text = "Also unrelated: " + " ".join(words[midpoint:])
    out, evidence = audit_expected_in_inputs(
        {"expected": _LONG_EXPECTED}, (first_half_text, second_half_text), enabled=True
    )
    assert "cancelled" not in out
    assert evidence is None
