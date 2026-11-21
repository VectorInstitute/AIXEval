"""run_cross_modal_hallucination_judge: image_path / evidence-specific cases.

Shared output/mutate/max_tokens/sealed-key cases live in test_judge_runner_wiring.
"""

from aixeval.judge.runner import run_cross_modal_hallucination_judge


class _FakeClient:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls = 0
        self.image_calls = 0
        self.image_paths_seen: list[str] = []

    @property
    def model(self) -> str:
        return "fake-model"

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        self.calls += 1
        return self._reply

    def complete_with_image(self, prompt: str, image_path: str, *, max_tokens: int = 1024) -> str:
        self.image_calls += 1
        self.image_paths_seen.append(image_path)
        return self._reply


def test_image_path_is_preferred_over_image_description() -> None:
    record = {
        "image_path": "/data/dog.jpg",
        "image_description": "A dog on a dock.",
        "output": "A cat on a dock.",
    }
    client = _FakeClient("SCORE: 1")

    new_record, attachment = run_cross_modal_hallucination_judge(record, client)

    assert new_record["cross_modal_hallucination_judge"] == 1.0
    assert attachment["status"] == "scored"
    assert client.image_calls == 1
    assert client.calls == 0
    assert client.image_paths_seen == ["/data/dog.jpg"]


def test_image_path_alone_is_used() -> None:
    record = {"image_path": "/data/dog.jpg", "output": "A dog on a dock."}
    client = _FakeClient("SCORE: 0")

    new_record, attachment = run_cross_modal_hallucination_judge(record, client)

    assert new_record["cross_modal_hallucination_judge"] == 0.0
    assert attachment["status"] == "scored"
    assert client.image_calls == 1


def test_missing_image_evidence_is_a_failed_attachment_no_client_call() -> None:
    record = {"output": "A cat on a dock."}
    client = _FakeClient("SCORE: 0")

    new_record, attachment = run_cross_modal_hallucination_judge(record, client)

    assert new_record == record
    assert attachment["status"] == "failed"
    assert attachment["reason"] == "record has no image_path or image_description to judge against"
    assert client.calls == 0
    assert client.image_calls == 0
