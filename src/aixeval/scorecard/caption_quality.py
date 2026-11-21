"""caption_quality raw score: CIDEr, TF-IDF n-gram cosine similarity against references.

Literature: Vedantam, Zitnick, and Parikh, CIDEr: Consensus-based
Image Description Evaluation, CVPR 2015 (SPICE is a scene-graph
alternative metric; see the declared gap below). Common per-dataset
benchmarks include MS-COCO and NoCaps.

Tier 1 (algorithmic), no new dependency: implemented directly in pure
Python (``re``/``collections.Counter``/``math``, all stdlib), the
same "no heavy library needed" spirit as ``f1``. For n-gram lengths
1-4, build a TF-IDF-weighted vector per text, cosine-similarity the
candidate against each reference, average over references, then
average the four n-gram-length scores.

Declared simplification, flagged honestly: real CIDEr computes IDF
(inverse document frequency) over the *entire* reference corpus for
the whole dataset, specifically to down-weight n-grams common across
many different images (like "a picture of"). This module can only
see the ``M`` references declared on *this one record*; there is no
whole-dataset corpus available to a per-record scorer. IDF here is
computed locally, from just those ``M`` references, with scikit-learn-
style smoothed IDF (``log((M+1)/(df+1)) + 1``, always strictly
positive). The ``+1`` after the log matters, not just the ``+1``
inside it: without it, a single reference (``M=1``, the common case)
gives every n-gram ``df=1``, so ``log((1+1)/(1+1)) == log(1) == 0``
for every term, zeroing the whole candidate/reference vectors and
scoring even an exact match as 0.0 (a real bug caught via live
testing, not a theoretical concern). This is a real, documented
divergence from the official metric, the same "declared variant, not
the official script" spirit already applied to ``bleu``/``rouge``/
``wer`` elsewhere in this toolkit, not silently passed off as the
real thing.

A second, related bug caught the same way: when text is shorter than
``n`` tokens, there are no n-grams of that length at all. If this
happens on *both* sides (candidate and every reference), that is not
a mismatch to penalize; there's simply no information at that
n-gram length. Treated as a match (1.0), the same "both empty -> 1.0"
convention ``f1.py``'s ``_token_f1`` already uses; without it, an
exact short-text match (e.g. ``"a dog"`` against itself) scored only
0.5, not 1.0, because the missing ``n=3``/``n=4`` terms fell through
to a degenerate 0.0 from comparing two empty vectors.

SPICE (a scene-graph F1 alternative to CIDEr) is NOT implemented
here: the official SPICE scorer needs a
dependency parser and semantic-role scene-graph construction
(typically run via Stanford CoreNLP, a heavy Java toolchain), a much
bigger lift than CIDEr's pure n-gram statistics. Left as a documented
gap, not attempted.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "caption_quality"
_PUNCTUATION_RE = re.compile(r"[^\w\s]")
_NGRAM_LENGTHS = (1, 2, 3, 4)


def score_caption_quality(record: dict[str, Any]) -> dict[str, Any]:
    """Score ``output`` (a candidate caption) against ``reference_captions``.

    Reads ``reference_captions`` (``M`` declared reference captions)
    and ``output``. Raw score is CIDEr, averaged over 1-4-gram TF-IDF
    cosine similarity (IDF computed locally from the ``M`` references;
    see the module docstring). Higher is better. Missing gold or
    output is N/A, never a silent 0.0.
    """
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("cancelled")

    references, references_ok = _optional_str_list(record, "reference_captions")
    output, output_ok = _optional_text(record, "output")
    if not references_ok or not output_ok:
        return _na("missing_evidence")
    if references is None:
        return _na("no_gold")
    if output is None:
        return _na("missing_output")

    raw_score = _cider(output, references)
    return _scored(raw_score)


def _cider(candidate: str, references: list[str]) -> float:
    candidate_tokens = _tokenize(candidate)
    reference_token_lists = [_tokenize(reference) for reference in references]
    scores = [_cider_for_n(candidate_tokens, reference_token_lists, n) for n in _NGRAM_LENGTHS]
    return sum(scores) / len(scores)


def _cider_for_n(candidate_tokens: list[str], reference_token_lists: list[list[str]], n: int) -> float:
    candidate_ngrams = Counter(_ngrams(candidate_tokens, n))
    reference_ngram_counts = [Counter(_ngrams(tokens, n)) for tokens in reference_token_lists]
    # Text shorter than n has no n-grams at all. If NEITHER side has
    # any (both candidate and every reference), there's no mismatch
    # to penalize, just no information at this n; treat as a match,
    # the same "both empty -> 1.0" convention f1.py's _token_f1 uses.
    # Catches a real bug: without this, an exact short-text match
    # (e.g. "a dog" == "a dog") scored only 0.5, not 1.0, because
    # n=3/4 have no n-grams on either side and fell through to a
    # degenerate 0.0 from cosine-of-two-empty-vectors.
    if not candidate_ngrams and all(not ngrams for ngrams in reference_ngram_counts):
        return 1.0
    document_count = len(reference_ngram_counts)
    document_frequency: Counter[tuple[str, ...]] = Counter()
    for ngrams in reference_ngram_counts:
        for ngram in ngrams:
            document_frequency[ngram] += 1

    def idf(ngram: tuple[str, ...]) -> float:
        # Smoothed IDF (+1 after the log, the same fix scikit-learn's
        # TfidfVectorizer uses): without it, a single reference (M=1,
        # the common case) gives every n-gram df=1, so
        # log((1+1)/(1+1)) == log(1) == 0 for every term, zeroing out
        # the whole vector, even for an exact candidate/reference
        # match. The +1 keeps idf strictly positive (minimum 1, for a
        # term in every reference) so cosine similarity still reduces
        # correctly to plain term-frequency matching when M is small.
        return math.log((document_count + 1) / (document_frequency.get(ngram, 0) + 1)) + 1.0

    candidate_vector = {ngram: count * idf(ngram) for ngram, count in candidate_ngrams.items()}
    similarities = [
        _cosine(candidate_vector, {ngram: count * idf(ngram) for ngram, count in ngrams.items()})
        for ngrams in reference_ngram_counts
    ]
    return sum(similarities) / len(similarities) if similarities else 0.0


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def _cosine(vector_a: dict[tuple[str, ...], float], vector_b: dict[tuple[str, ...], float]) -> float:
    common = set(vector_a) & set(vector_b)
    dot = sum(vector_a[key] * vector_b[key] for key in common)
    norm_a = math.sqrt(sum(value * value for value in vector_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vector_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _tokenize(text: str) -> list[str]:
    return _PUNCTUATION_RE.sub(" ", text.lower()).split()


def _optional_str_list(record: dict[str, Any], key: str) -> tuple[list[str] | None, bool]:
    """Return a non-empty string list and whether the field type is usable."""
    if key not in record or record[key] is None:
        return None, True
    value = record[key]
    if not isinstance(value, list) or not value:
        return None, False
    if not all(isinstance(item, str) and item for item in value):
        return None, False
    return value, True


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
    return apply_grade({"metric_id": METRIC_ID, "status": "scored", "raw_score": min(1.0, max(0.0, raw_score))})


def _na(reason: str) -> dict[str, Any]:
    return {"metric_id": METRIC_ID, "status": "na", "na_reason": require_na_reason(reason)}


def print_caption_quality_examples() -> None:
    """Print how output/reference_captions become a CIDEr raw score and grade."""
    print("caption_quality asks: how close is output to the reference captions (CIDEr)?")
    print("  TF-IDF n-gram (1-4) cosine similarity, averaged over references and n.")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        (
            "Exact match to one of the references.",
            {"output": "a dog running on the beach", "reference_captions": ["a dog running on the beach"]},
        ),
        (
            "Close paraphrase of the references.",
            {
                "output": "a dog runs along the beach",
                "reference_captions": ["a dog running on the beach", "a dog runs on the sand"],
            },
        ),
        (
            "Unrelated caption.",
            {"output": "a cat sleeping on a couch", "reference_captions": ["a dog running on the beach"]},
        ),
        ("No reference captions on the record.", {"output": "a dog running on the beach"}),
        (
            "Case was cancelled (for example an audit stop).",
            {
                "output": "a dog running on the beach",
                "reference_captions": ["a dog running on the beach"],
                "cancelled": True,
            },
        ),
    )
    for story, record in examples:
        result = score_caption_quality(record)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_caption_quality_examples()
