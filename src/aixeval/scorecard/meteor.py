"""meteor raw score: NLTK METEOR (stem + synonym aware) on [0, 1].

Literature: Banerjee and Lavie, METEOR: An Automatic Metric for MT
Evaluation with Improved Correlation with Human Judgments, ACL 2005
Workshop on Intrinsic and Extrinsic Evaluation Measures.

Tier 1 (library-backed), but opt-in, unlike bleu/rouge/wer/f1: this
metric is deliberately NOT in
``aixeval.scorecard.metrics.POSTHOC_METRICS`` and
``score_metrics()`` never calls it automatically. Call
``score_meteor`` directly when you want it. Two independent reasons:

1. **Real dependency, smaller than BERTScore's but real:** synonym
   matching needs NLTK's WordNet corpus, which ``pip install nltk``
   does NOT bundle; it's a separate one-time download. Exact-text
   pairs score fine without it (stem matching alone succeeds); the
   moment two different words need a synonym check, nltk raises
   ``LookupError`` until the corpus is fetched. Setup:
   ``uv run python -c "import nltk; nltk.download('wordnet');
   nltk.download('omw-1.4')"``.

2. **Known limitation, confirmed by live testing here:** METEOR's
   fragmentation-penalty term scores short/single-token exact matches
   harshly: ``meteor_score([["Paris"]], ["Paris"])`` returns
   ``0.5``, not ~1.0, even though it's a perfect match (longer exact
   matches like "the cat sat on the mat" score ~0.998; only short
   answers get hit). Because ``capability``'s card grade is the
   *minimum* across every scored capability metric, wiring this into
   ``score_metrics()`` would drag a perfect short answer's capability
   grade down even though ``task_success`` correctly grades it 4.
   This is a genuine property of METEOR's formula, not a bug in this
   integration, but it means METEOR is a poor fit for short-answer
   QA-style records specifically, unlike BLEU/ROUGE/WER/F1/BERTScore.
   Better suited to longer, sentence-or-paragraph-length text.

Unit tests use an injectable ``scorer`` so the hermetic unit suite
never needs the WordNet corpus.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "meteor"

# (hypothesis, reference) -> raw METEOR score.
MeteorScorerFn = Callable[[str, str], float]


def score_meteor(record: dict[str, Any], *, scorer: MeteorScorerFn | None = None) -> dict[str, Any]:
    """Score stem/synonym-aware overlap of ``output`` against ``expected``.

    Reads ``expected`` (reference) and ``output`` (hypothesis): the
    same fields ``task_success`` reads, not a separate reference field.
    Missing gold or output is N/A, never a silent 0.0.

    Parameters
    ----------
    record
        The sealed record.
    scorer
        Injectable ``(hypothesis, reference) -> raw_score`` callable.
        Tests inject a fake here so the unit suite never needs the
        WordNet corpus. Defaults to a real ``nltk`` METEOR call.

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
    raw_score = min(1.0, max(0.0, run(output, expected)))
    return _scored(raw_score)


def _default_scorer(hypothesis: str, reference: str) -> float:
    """Real nltk METEOR call. Needs the WordNet corpus for synonym matching."""
    from nltk.translate.meteor_score import meteor_score  # noqa: PLC0415 (keeps corpus lookup lazy)

    return meteor_score([reference.split()], hypothesis.split())


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


def _demo_scorer(hypothesis: str, reference: str) -> float:
    """Fake, deterministic scorer for the walkthrough: no WordNet lookup."""
    return 1.0 if hypothesis.strip().lower() == reference.strip().lower() else 0.4


def print_meteor_examples() -> None:
    """Print how expected/output become a METEOR raw score, using a fake scorer.

    Does not call the real nltk METEOR: that needs the WordNet
    corpus (see module docstring for the one-time setup command).
    Call ``score_meteor(record)`` with no ``scorer=`` override to use
    the real metric once the corpus is downloaded.
    """
    print("meteor asks: how much stem/synonym-aware overlap does output have with expected?")
    print("  expected = reference text. output = what the model said.")
    print("  This demo uses a fake scorer to avoid requiring the WordNet corpus.")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("Exact match text.", {"expected": "the cat sat on the mat", "output": "the cat sat on the mat"}),
        ("Different text.", {"expected": "the cat sat on the mat", "output": "quarterly revenue rose"}),
        ("No gold answer on the record.", {"output": "yes"}),
        ("Gold exists, but the model gave no answer.", {"expected": "yes"}),
    )
    for story, record in examples:
        result = score_meteor(record, scorer=_demo_scorer)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_meteor_examples()
