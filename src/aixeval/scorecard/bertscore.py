"""bertscore raw score: DeBERTa-backed contextual-embedding F1 on [0, 1].

Literature: Zhang et al., BERTScore: Evaluating Text Generation with BERT,
ICLR 2020.

Tier 1 (library-backed), but opt-in, unlike bleu/rouge/wer: this metric is
deliberately NOT in ``aixeval.scorecard.metrics.POSTHOC_METRICS`` and
``score_metrics()`` never calls it automatically. Call ``score_bertscore``
directly when you want it.

Heavy dependency, deliberately contained: importing this module does
NOT pull in torch/transformers. That import, and the first real call's
model download (``microsoft/deberta-xlarge-mnli``, several GB), only
happens the first time ``score_bertscore`` actually runs, never at
``import aixeval.scorecard`` time.

Acknowledged and accepted, not hidden: DeBERTa-xlarge-mnli is the
bert-score maintainers' own recommended backbone for best accuracy:
stronger correlation with human judgment than the package's own
``roberta-large`` default for English, at real cost (torch,
transformers, a large one-time download, slower than BLEU/ROUGE/WER).

Known limitation, confirmed by live testing here: BERTScore is
insensitive to single-entity / factual swaps. "The capital of France
is Paris" vs. "...is Lyon" scored 0.94 raw, higher than a genuine
correct paraphrase, because changing one content word barely moves
the sentence embedding. This is a documented property of contextual-
embedding metrics generally, not a bug in this integration. Do not use
bertscore alone to grade single-fact correctness; that is what
task_success (exact match) is for. bertscore is for paraphrase-
tolerant similarity on longer, more free-form text.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "bertscore"
DEFAULT_MODEL_TYPE = "microsoft/deberta-xlarge-mnli"

# (cands, refs) -> (P, R, F1), matching bert_score.score's shape.
BertScoreFn = Callable[[list[str], list[str]], tuple[Any, Any, Any]]


def score_bertscore(record: dict[str, Any], *, scorer: BertScoreFn | None = None) -> dict[str, Any]:
    """Score contextual-embedding F1 of ``output`` against ``expected``.

    Reads ``expected`` (reference) and ``output`` (hypothesis): the
    same fields ``task_success`` reads, not a separate reference field.
    Missing gold or output is N/A, never a silent 0.0.

    Parameters
    ----------
    record
        The sealed record.
    scorer
        Injectable ``(cands, refs) -> (P, R, F1)`` callable. Tests
        inject a fake here so the unit suite never downloads a model.
        Defaults to a real ``bert_score.score`` call against
        ``DEFAULT_MODEL_TYPE``, imported lazily at call time.

    Returns
    -------
    dict[str, Any]
        The usual scored/N/A row shape.
    """
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("cancelled")

    expected, expected_ok = _optional_text(record, "expected")
    output, output_ok = _optional_text(record, "output")
    if not expected_ok or not output_ok:
        return _na("missing_evidence")
    if expected is None:
        return _na("no_gold")
    if output is None:
        return _na("missing_output")

    run = scorer if scorer is not None else _default_scorer
    _precision, _recall, f1 = run([output], [expected])
    raw_score = min(1.0, max(0.0, float(f1[0])))
    return _scored(raw_score)


_UNBOUNDED_SENTINEL = 100_000
# DeBERTa-xlarge-mnli's real limit (its own config.json: max_position_embeddings).
_DEBERTA_MAX_LENGTH = 512


def _default_scorer(cands: list[str], refs: list[str]) -> tuple[Any, Any, Any]:
    """Real bert_score.score call. Imports torch/transformers here, not at load.

    Works around a real upstream bug: the DeBERTa tokenizer ships with no
    ``model_max_length``, so transformers defaults it to a sentinel meaning
    "unbounded" (``int(1e30)``). ``bert_score`` (unmaintained since well
    before transformers' Rust-backed tokenizers became the only path) passes
    that sentinel straight into ``tokenizer.encode(..., truncation=True)``,
    which now raises ``OverflowError: int too big to convert``: the Rust
    binding cannot represent it. Patching the tokenizer's declared max
    length down to DeBERTa's real limit avoids the crash without touching
    bert_score's own package.
    """
    import sys  # noqa: PLC0415

    import bert_score.score  # noqa: F401, PLC0415 (heavy import; used via sys.modules below)

    # bert_score/__init__.py does `from .score import score`, which rebinds
    # the `bert_score.score` attribute to that function, so `import
    # bert_score.score as x` would silently hand us the function, not the
    # module. Go through sys.modules for the real module object.
    bert_score_module: Any = sys.modules["bert_score.score"]
    original_get_tokenizer = bert_score_module.get_tokenizer

    def _get_tokenizer_with_bounded_max_length(model_type: str, use_fast: bool = False) -> Any:  # noqa: FBT001, FBT002
        tokenizer = original_get_tokenizer(model_type, use_fast)
        if tokenizer.model_max_length > _UNBOUNDED_SENTINEL:
            tokenizer.model_max_length = _DEBERTA_MAX_LENGTH
        return tokenizer

    bert_score_module.get_tokenizer = _get_tokenizer_with_bounded_max_length
    try:
        return bert_score_module.score(cands, refs, model_type=DEFAULT_MODEL_TYPE)
    finally:
        bert_score_module.get_tokenizer = original_get_tokenizer


def _optional_text(record: dict[str, Any], key: str) -> tuple[str | None, bool]:
    """Return stripped text and whether the field type is usable."""
    if key not in record or record[key] is None:
        return None, True
    value = record[key]
    if not isinstance(value, str):
        return None, False
    stripped = value.strip()
    return (stripped or None), True


def _scored(raw_score: float) -> dict[str, Any]:
    return apply_grade({"metric_id": METRIC_ID, "status": "scored", "raw_score": raw_score})


def _na(reason: str) -> dict[str, Any]:
    return {"metric_id": METRIC_ID, "status": "na", "na_reason": require_na_reason(reason)}


def _demo_scorer(cands: list[str], refs: list[str]) -> tuple[list[float], list[float], list[float]]:
    """Fake, deterministic scorer for the walkthrough: no download, no torch."""
    same = [1.0 if c.strip().lower() == r.strip().lower() else 0.2 for c, r in zip(cands, refs, strict=True)]
    return same, same, same


def print_bertscore_examples() -> None:
    """Print how expected/output become a raw score, using a fake scorer.

    Does not call the real bert_score: that needs torch + transformers
    and downloads microsoft/deberta-xlarge-mnli (several GB) on first
    use. Call ``score_bertscore(record)`` with no ``scorer=`` override
    to use the real model.
    """
    print("bertscore asks: how similar are output and expected in embedding space?")
    print("  expected = reference text. output = what the model said.")
    print(f"  Real scorer: bert_score.score, model_type={DEFAULT_MODEL_TYPE!r}.")
    print("  This demo uses a fake scorer to avoid an unprompted multi-GB download.")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("Exact match text.", {"expected": "the cat sat on the mat", "output": "the cat sat on the mat"}),
        ("Different text.", {"expected": "the cat sat on the mat", "output": "quarterly revenue rose"}),
        ("No gold answer on the record.", {"output": "yes"}),
        ("Gold exists, but the model gave no answer.", {"expected": "yes"}),
    )
    for story, record in examples:
        result = score_bertscore(record, scorer=_demo_scorer)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_bertscore_examples()
