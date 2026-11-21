"""run_grounding_judge: evidence extraction / excerpt cases.

Shared output/mutate/max_tokens/sealed-key cases live in test_judge_runner_wiring.
"""

from aixeval.judge.runner import EVIDENCE_EXCERPT_CHARS, run_grounding_judge


class _FakeClient:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls = 0

    @property
    def model(self) -> str:
        return "fake-model"

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        self.calls += 1
        return self._reply


def _no_evidence(_payload: dict) -> str | None:
    return None


def _fixed_evidence(text: str):
    def extractor(payload: dict) -> str | None:
        return text

    return extractor


def test_no_evidence_is_a_failed_attachment_no_client_call() -> None:
    record = {"output": "Mexico sales grew."}
    client = _FakeClient("SCORE: 1")

    new_record, attachment = run_grounding_judge({}, record, client, _no_evidence)

    assert "grounding_judge" not in new_record
    assert new_record == record
    assert attachment["status"] == "failed"
    assert attachment["reason"] == "no evidence extracted from the dump"
    assert "evidence_excerpt" not in attachment
    assert "raw_reply" not in attachment
    assert client.calls == 0


def test_long_evidence_is_excerpted() -> None:
    record = {"output": "x"}
    client = _FakeClient("SCORE: 1")
    long_evidence = "a" * (EVIDENCE_EXCERPT_CHARS + 50)

    _new_record, attachment = run_grounding_judge({}, record, client, _fixed_evidence(long_evidence))

    assert len(attachment["evidence_excerpt"]) == EVIDENCE_EXCERPT_CHARS + 1  # + ellipsis
    assert attachment["evidence_excerpt"].endswith("…")


def test_short_evidence_is_not_truncated() -> None:
    record = {"output": "x"}
    client = _FakeClient("SCORE: 1")

    _new_record, attachment = run_grounding_judge({}, record, client, _fixed_evidence("short"))

    assert attachment["evidence_excerpt"] == "short"


def test_extract_evidence_receives_the_raw_payload() -> None:
    seen = []

    def extractor(payload: dict) -> str | None:
        seen.append(payload)
        return "evidence"

    record = {"output": "x"}
    client = _FakeClient("SCORE: 1")
    payload = {"some": "dump"}

    run_grounding_judge(payload, record, client, extractor)

    assert seen == [payload]
