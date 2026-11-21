"""Audit after ingest: gold overlapping declared input texts cancels the case.

Matching is windowed-n-gram overlap, not a single exact-substring check:
the standard technique the data-contamination-detection literature uses
(GPT-3/GPT-4's own leak screens are both variants of this), chosen over
embeddings or an LLM judge deliberately: it stays free and deterministic,
at the cost of still missing a true paraphrase (same ceiling the field's
own methods have, not a gap unique to this implementation). For gold
text at or under ``WINDOW_WORDS`` words, this is exactly a full-string
containment check (one window, the whole text); it degrades to the
prior exact-match behavior for a short gold answer, and only starts
catching a *partial* leak (one word changed, reformatted, mid-leak cut)
once gold is longer than one window.
"""

from __future__ import annotations

from typing import Any

from aixeval.ingest.record import validate_record


# Declared policy. Shorter gold (yes, A) is too common in ordinary prose.
MIN_EXPECTED_LEN = 4

# Declared policy, not a literature-proven optimum. 8 words sits at the
# short end of GPT-3's 13-gram / GPT-4's ~50-character screens, chosen
# smaller because this toolkit's own gold answers skew short (a lot of
# them are a single word or short phrase), and a too-large window would
# never have more than one window to begin with, degrading every short
# answer back to exact-match with no partial-overlap benefit at all.
WINDOW_WORDS = 8

# Declared policy: at least half of gold's windows must turn up verbatim
# in the SAME input text before this counts as a leak. Checked per input
# text, not summed across all of them: unrelated input texts that each
# happen to contain a few of gold's words should never add up to a hit.
OVERLAP_THRESHOLD = 0.5

# How much of the matched input text to quote back in the evidence string.
EVIDENCE_EXCERPT_CHARS = 300


def audit_expected_in_inputs(
    record: dict[str, Any],
    input_texts: list[str] | tuple[str, ...],
    *,
    enabled: bool = False,
) -> tuple[dict[str, Any], str | None]:
    """If gold overlaps enough of any single input text, mark leak and cancel.

    Off by default. Does not open files. Does not call a model.
    A hit sets ``contaminated``, ``cancelled``, and
    ``evidence_status="cancelled"`` on the returned record: every
    scorer then reports N/A uniformly, which on its own gives no way to
    tell *why*. The second return value is that why: a human-readable
    explanation of the overlap fraction and which input text it matched
    (truncated), kept out of the record itself rather than added as a
    new field, the same reason a live judge's rationale is returned
    alongside the record instead of written into it. ``None`` when
    there was no hit, or the audit never ran (off, no ``expected``,
    gold too short).

    Returns
    -------
    tuple[dict[str, Any], str | None]
        The (possibly cancelled) record, and the evidence string or
        ``None``.
    """
    out = dict(record)
    if not enabled:
        return validate_record(out), None

    expected = record.get("expected")
    if not isinstance(expected, str):
        return validate_record(out), None
    needle = expected.strip().casefold()
    if len(needle) < MIN_EXPECTED_LEN:
        return validate_record(out), None

    windows = _windows(needle, WINDOW_WORDS)
    for raw in input_texts:
        if not isinstance(raw, str):
            continue
        haystack = raw.strip().casefold()
        fraction = _overlap_fraction(windows, haystack)
        if fraction >= OVERLAP_THRESHOLD:
            out["contaminated"] = True
            out["cancelled"] = True
            out["evidence_status"] = "cancelled"
            return validate_record(out), _evidence(fraction, raw)
    return validate_record(out), None


def _windows(text: str, size: int) -> list[str]:
    """Split ``text`` into overlapping ``size``-word windows.

    Text at or under ``size`` words returns ``[text]``: a single
    window spanning the whole string, so matching reduces to exact
    containment for short gold.
    """
    words = text.split()
    if len(words) <= size:
        return [text]
    return [" ".join(words[i : i + size]) for i in range(len(words) - size + 1)]


def _overlap_fraction(windows: list[str], haystack: str) -> float:
    """Return the share of ``windows`` found verbatim in ``haystack``."""
    if not windows:
        return 0.0
    hits = sum(1 for window in windows if window in haystack)
    return hits / len(windows)


def _evidence(fraction: float, matched_input_text: str) -> str:
    """Human-readable explanation of a contamination hit."""
    excerpt = matched_input_text.strip()
    if len(excerpt) > EVIDENCE_EXCERPT_CHARS:
        excerpt = excerpt[:EVIDENCE_EXCERPT_CHARS] + "..."
    return f"{fraction:.0%} of gold's {WINDOW_WORDS}-word windows found verbatim in this input text: {excerpt!r}"


def print_audit_examples() -> None:
    """Print how a gold-overlap hit cancels a record."""
    print("audit_expected_in_inputs is off by default. Min gold length is 4 chars.")
    print(f"Windowed overlap: {WINDOW_WORDS}-word windows, cancels at >= {OVERLAP_THRESHOLD:.0%} found verbatim.")
    print()
    record = {"expected": "Paris", "output": "Paris"}
    texts = ("Question: Capital? The answer is Paris.",)
    print(f"  record: {record}")
    print(f"  texts:  {texts}")
    print(f"  off:    {audit_expected_in_inputs(record, texts)}")
    print(f"  on:     {audit_expected_in_inputs(record, texts, enabled=True)}")
    print()
    long_expected = (
        "the clause is enforceable against sophisticated commercial parties because it was "
        "conspicuously disclosed in the body of the contract and not buried in fine print"
    )
    long_record = {"expected": long_expected}
    partial_leak = (
        "Context: "
        + " ".join(long_expected.split()[: int(len(long_expected.split()) * 0.7)])
        + " -- something totally different and unrelated here instead.",
    )
    out, evidence = audit_expected_in_inputs(long_record, partial_leak, enabled=True)
    print("A longer gold answer, only the first 70% leaked verbatim (the rest never appears):")
    print(f"  record:   {long_record}")
    print(f"  texts:    {partial_leak}")
    print(f"  out:      {out}")
    print(f"  evidence: {evidence}")
    print("  (the old exact-substring check would have missed this: the full string never appears)")


if __name__ == "__main__":
    print_audit_examples()
