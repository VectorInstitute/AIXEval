"""run_tool_call_accuracy_check: dump + record -> new record + attachment. No LLM.

Precision/recall F1 over {tool, args} pairs, comparing the dump's
actual tool calls against the record's declared gold_tool_calls.
Never mutates the input record. Always returns an attachment on
success or failure.
"""

import pytest

from aixeval.judge.runner import run_tool_call_accuracy_check


def _fixed_tool_calls(calls: list[dict] | None):
    def extractor(payload: dict) -> list[dict] | None:
        return calls

    return extractor


def test_exact_match_scores_one() -> None:
    record = {"gold_tool_calls": [{"tool": "get_weather", "args": {"city": "Paris"}}]}
    payload = {"observations": "ignored by the fixed extractor"}

    new_record, attachment = run_tool_call_accuracy_check(
        payload, record, _fixed_tool_calls([{"tool": "get_weather", "args": {"city": "Paris"}}])
    )

    assert new_record["tool_call_accuracy_judge"] == 1.0
    assert attachment["status"] == "scored"
    assert "model" not in attachment
    assert "prompt_version" not in attachment
    assert "raw_reply" not in attachment
    assert "timestamp" in attachment
    assert "evidence" not in attachment  # perfect match, nothing to explain


def test_does_not_mutate_the_input_record() -> None:
    record = {"gold_tool_calls": [{"tool": "get_weather", "args": {}}]}
    original = dict(record)

    run_tool_call_accuracy_check({}, record, _fixed_tool_calls([{"tool": "get_weather", "args": {}}]))

    assert record == original


def test_missing_required_call_lowers_the_score() -> None:
    record = {
        "gold_tool_calls": [
            {"tool": "get_weather", "args": {"city": "Paris"}},
            {"tool": "convert_temperature", "args": {"from": "C", "to": "F"}},
        ]
    }

    new_record, attachment = run_tool_call_accuracy_check(
        {}, record, _fixed_tool_calls([{"tool": "get_weather", "args": {"city": "Paris"}}])
    )

    # 1 correct of 2 gold, 1 correct of 1 actual -> precision 1.0, recall 0.5, F1 2/3
    assert new_record["tool_call_accuracy_judge"] == pytest.approx(2 / 3)
    assert "missing gold call(s): convert_temperature" in attachment["evidence"]
    assert "extra/unexpected" not in attachment["evidence"]


def test_spurious_extra_call_lowers_the_score() -> None:
    record = {"gold_tool_calls": [{"tool": "get_weather", "args": {"city": "Paris"}}]}

    new_record, attachment = run_tool_call_accuracy_check(
        {},
        record,
        _fixed_tool_calls([{"tool": "get_weather", "args": {"city": "Paris"}}, {"tool": "send_email", "args": {}}]),
    )

    # 1 correct of 1 gold (recall 1.0), 1 correct of 2 actual (precision 0.5) -> F1 2/3
    assert new_record["tool_call_accuracy_judge"] == pytest.approx(2 / 3)
    assert "extra/unexpected call(s): send_email" in attachment["evidence"]
    assert "missing gold" not in attachment["evidence"]


def test_no_tool_calls_made_scores_zero_against_nonempty_gold() -> None:
    record = {"gold_tool_calls": [{"tool": "get_weather", "args": {}}]}

    new_record, attachment = run_tool_call_accuracy_check({}, record, _fixed_tool_calls(None))

    assert new_record["tool_call_accuracy_judge"] == 0.0
    assert attachment["status"] == "scored"


def test_no_tool_calls_expected_and_none_made_scores_one() -> None:
    record = {"gold_tool_calls": []}

    new_record, attachment = run_tool_call_accuracy_check({}, record, _fixed_tool_calls(None))

    assert new_record["tool_call_accuracy_judge"] == 1.0
    assert attachment["status"] == "scored"


def test_missing_gold_tool_calls_is_a_failed_attachment() -> None:
    record: dict = {}

    new_record, attachment = run_tool_call_accuracy_check(
        {}, record, _fixed_tool_calls([{"tool": "get_weather", "args": {}}])
    )

    assert new_record == record
    assert "tool_call_accuracy_judge" not in new_record
    assert attachment["status"] == "failed"
    assert attachment["reason"] == "record has no gold_tool_calls to compare against"


def test_bad_type_gold_tool_calls_is_a_failed_attachment() -> None:
    record = {"gold_tool_calls": "not a list"}

    new_record, attachment = run_tool_call_accuracy_check({}, record, _fixed_tool_calls(None))

    assert new_record == record
    assert attachment["status"] == "failed"


def test_args_order_does_not_matter_for_a_match() -> None:
    record = {"gold_tool_calls": [{"tool": "search", "args": {"a": 1, "b": 2}}]}

    new_record, _attachment = run_tool_call_accuracy_check(
        {}, record, _fixed_tool_calls([{"tool": "search", "args": {"b": 2, "a": 1}}])
    )

    assert new_record["tool_call_accuracy_judge"] == 1.0


def test_extract_tool_calls_receives_the_raw_payload() -> None:
    seen = []

    def extractor(payload: dict) -> list[dict] | None:
        seen.append(payload)
        return [{"tool": "get_weather", "args": {}}]

    record = {"gold_tool_calls": [{"tool": "get_weather", "args": {}}]}
    payload = {"some": "dump"}

    run_tool_call_accuracy_check(payload, record, extractor)

    assert seen == [payload]


def test_unknown_record_key_is_rejected_by_validate_record() -> None:
    record = {"gold_tool_calls": [], "not_a_real_field": "oops"}

    with pytest.raises(ValueError, match="Unknown sealed-record field"):
        run_tool_call_accuracy_check({}, record, _fixed_tool_calls(None))


def test_no_tool_calls_expected_and_none_made_has_no_evidence() -> None:
    record = {"gold_tool_calls": []}

    _new_record, attachment = run_tool_call_accuracy_check({}, record, _fixed_tool_calls(None))

    assert "evidence" not in attachment


def test_duplicate_extra_call_is_named_once_not_per_duplicate() -> None:
    record = {"gold_tool_calls": [{"tool": "get_weather", "args": {"city": "Paris"}}]}

    _new_record, attachment = run_tool_call_accuracy_check(
        {},
        record,
        _fixed_tool_calls(
            [
                {"tool": "get_weather", "args": {"city": "Paris"}},
                {"tool": "get_weather", "args": {"city": "Paris"}},
            ]
        ),
    )

    assert attachment["evidence"].count("get_weather") == 1
