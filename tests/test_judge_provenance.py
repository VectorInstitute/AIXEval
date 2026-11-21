"""judge_provenance helpers: merge onto a record, read models back."""

from aixeval.judge.provenance import merge_judge_provenance, models_from_provenance


class _FakeClient:
    def __init__(self, model: str = "fake-model") -> None:
        self._model = model

    @property
    def model(self) -> str:
        return self._model


def test_merge_writes_model_and_prompt_version() -> None:
    record = {"output": "Paris"}
    updated = merge_judge_provenance(record, "grounding", _FakeClient(), "grounding-v1")
    assert updated["judge_provenance"]["grounding"] == {"model": "fake-model", "prompt_version": "grounding-v1"}
    assert record == {"output": "Paris"}


def test_merge_keeps_other_metrics() -> None:
    record = {
        "judge_provenance": {"toxicity": {"model": "old", "prompt_version": "toxicity-judge-v1"}},
    }
    updated = merge_judge_provenance(record, "grounding", _FakeClient("new"), "grounding-v1")
    assert updated["judge_provenance"]["toxicity"]["model"] == "old"
    assert updated["judge_provenance"]["grounding"]["model"] == "new"


def test_models_from_provenance_dedupes_in_first_seen_order() -> None:
    record = {
        "judge_provenance": {
            "grounding": {"model": "a", "prompt_version": "grounding-v1"},
            "toxicity": {"model": "b", "prompt_version": "toxicity-judge-v1"},
            "refusal": {"model": "a", "prompt_version": "refusal-v1"},
        }
    }
    assert models_from_provenance(record) == ("a", "b")
    assert models_from_provenance(None) == ()
    assert models_from_provenance({}) == ()
