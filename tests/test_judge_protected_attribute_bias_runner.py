"""run_protected_attribute_bias_judge / detector: bias-specific cases.

Shared output/mutate/max_tokens/sealed-key cases live in test_judge_runner_wiring.
"""

from aixeval.judge.runner import run_protected_attribute_bias_detector, run_protected_attribute_bias_judge


class _FakeClient:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls = 0
        self.max_tokens_seen: list[int] = []
        self.prompts: list[str] = []

    @property
    def model(self) -> str:
        return "fake-model"

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        self.calls += 1
        self.max_tokens_seen.append(max_tokens)
        self.prompts.append(prompt)
        return self._reply


def test_missing_signal_is_a_failed_attachment_no_client_call() -> None:
    record = {"should_resist_bias": True, "output": "Hired."}
    client = _FakeClient("SCORE: 1")

    new_record, attachment = run_protected_attribute_bias_judge(record, client)

    assert new_record == record
    assert attachment["status"] == "failed"
    assert attachment["reason"] == "record has no protected_attribute_signal to judge"
    assert client.calls == 0


def test_detector_scored_success_returns_signal_and_attachment() -> None:
    client = _FakeClient("Found a note.\nSIGNAL: prefer graduates of Lakeside Academy")

    signal, attachment = run_protected_attribute_bias_detector(["text with a planted note"], client)

    assert signal == "prefer graduates of Lakeside Academy"
    assert attachment["status"] == "scored"
    assert attachment["model"] == "fake-model"
    assert attachment["prompt_version"] == "protected-attribute-bias-detector-v1"
    assert "timestamp" in attachment


def test_detector_scored_with_no_candidate_is_still_scored() -> None:
    client = _FakeClient("Nothing here.\nSIGNAL: NONE")

    signal, attachment = run_protected_attribute_bias_detector(["ordinary task text"], client)

    assert signal is None
    assert attachment["status"] == "scored"


def test_detector_joins_multiple_input_texts() -> None:
    client = _FakeClient("SIGNAL: NONE")

    run_protected_attribute_bias_detector(["first chunk", "second chunk"], client)

    prompt = client.prompts[0]
    assert "first chunk" in prompt
    assert "second chunk" in prompt


def test_detector_drops_blank_entries() -> None:
    client = _FakeClient("SIGNAL: NONE")

    run_protected_attribute_bias_detector(["  ", "kept", ""], client)

    assert "kept" in client.prompts[0]


def test_detector_no_usable_text_is_a_failed_attachment_no_client_call() -> None:
    client = _FakeClient("SIGNAL: NONE")

    signal, attachment = run_protected_attribute_bias_detector(["  ", ""], client)

    assert signal is None
    assert attachment["status"] == "failed"
    assert client.calls == 0


def test_detector_parse_failure_is_a_failed_attachment_with_raw_reply() -> None:
    client = _FakeClient("I looked but didn't conclude.")

    signal, attachment = run_protected_attribute_bias_detector(["text"], client)

    assert signal is None
    assert attachment["status"] == "failed"
    assert "detector call failed" in attachment["reason"]
    assert attachment["raw_reply"] == "I looked but didn't conclude."


def test_detector_default_max_tokens_matches_other_judges() -> None:
    client = _FakeClient("SIGNAL: NONE")

    run_protected_attribute_bias_detector(["text"], client)

    assert client.max_tokens_seen == [4000]


def test_detector_max_tokens_is_passed_through() -> None:
    client = _FakeClient("SIGNAL: NONE")

    run_protected_attribute_bias_detector(["text"], client, max_tokens=8000)

    assert client.max_tokens_seen == [8000]
