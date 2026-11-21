"""clip_score raw score: CLIP image-text embedding similarity, on roughly [0, 1].

Literature: Hessel, Holtzman, Forbes, Bras, and Choi, CLIPScore: A
Reference-free Evaluation Metric for Image Captioning, EMNLP 2021
(verified via search, not guessed). The real formula:

    CLIPScore(c, v) = 2.5 * max(cos(c, v), 0)

where ``c`` is the candidate text's CLIP embedding and ``v`` is the
image's CLIP embedding. The paper's own authors chose the 2.5 factor
specifically to stretch the distribution to roughly span [0, 1] (it
can occasionally exceed 1, clipped here, same clipping discipline as
``wer``/``latency_vs_budget``); this toolkit uses the real formula
unmodified, not a rescaled variant.

Tier 1 (library-backed), but opt-in like ``bertscore``/``toxicity``'s
classifier path: this is the first metric in this toolkit that loads
a real image file. Needs ``transformers`` (already a transitive
dependency of ``bert-score``, no new hard dependency) and ``Pillow``
(already present transitively too) for the CLIP model and image
loading; downloads ``openai/clip-vit-base-patch32`` (~600MB) on first
real use. Deliberately NOT auto-wired: same heavy-dependency,
one-time-download profile as ``bertscore``. Call ``score_clip_score``
directly when you want it.
"""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "clip_score"
DEFAULT_MODEL = "openai/clip-vit-base-patch32"
RESCALE_FACTOR = 2.5

# (text, image_path) -> raw CLIPScore (already rescaled by 2.5, roughly [0, 1]).
ClipScorer = Callable[[str, str], float]


def score_clip_score(record: dict[str, Any], *, scorer: ClipScorer | None = None) -> dict[str, Any]:
    """Score CLIP embedding similarity between ``output`` and ``image_path``.

    Reference-free: reads ``output`` (candidate text) and
    ``image_path`` (a file path to the image), no gold caption
    needed. Missing path or output is N/A, never a silent 0.0.

    Parameters
    ----------
    record
        The sealed record.
    scorer
        Injectable ``(text, image_path) -> raw CLIPScore`` callable.
        Tests inject a fake here so the unit suite never downloads a
        model or touches a real image file. Defaults to a real CLIP
        call via ``transformers``, imported lazily at call time.

    Returns
    -------
    dict[str, Any]
        The usual scored/N/A row shape. Higher is better.
    """
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("cancelled")

    output, output_ok = _optional_text(record, "output")
    image_path, image_path_ok = _optional_text(record, "image_path")
    if not output_ok or not image_path_ok:
        return _na("missing_evidence")
    if image_path is None:
        return _na("no_gold")
    if output is None:
        return _na("missing_output")

    run = scorer if scorer is not None else _default_scorer
    raw_score = float(run(output, image_path))
    if not isfinite(raw_score):
        return _na("missing_evidence")
    raw_score = min(1.0, max(0.0, raw_score))
    return _scored(raw_score)


_model_cache: Any = None
_processor_cache: Any = None


def _default_scorer(text: str, image_path: str) -> float:
    """Real CLIPScore via ``transformers``/``Pillow``. Imports here, not at load.

    Caches the model and processor across calls in this process --
    the model load itself (not just the one-time download) is
    expensive enough to matter once more than a handful of records
    are scored.
    """
    global _model_cache, _processor_cache  # noqa: PLW0603 (process-local model cache)
    if _model_cache is None:
        import torch  # noqa: PLC0415 (heavy import, deferred to first real call)
        from PIL import Image  # noqa: PLC0415
        from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415

        _model_cache = CLIPModel.from_pretrained(DEFAULT_MODEL)
        _processor_cache = CLIPProcessor.from_pretrained(DEFAULT_MODEL)
    else:
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

    image = Image.open(image_path).convert("RGB")
    inputs = _processor_cache(text=[text], images=[image], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = _model_cache(**inputs)
    text_embed = outputs.text_embeds[0]
    image_embed = outputs.image_embeds[0]
    cosine = torch.nn.functional.cosine_similarity(text_embed, image_embed, dim=0).item()
    return RESCALE_FACTOR * max(cosine, 0.0)


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


def _demo_scorer(text: str, image_path: str) -> float:
    """Fake, deterministic scorer for the walkthrough: no download, no real image."""
    return 0.9 if "dog" in text.lower() and "dog" in image_path.lower() else 0.2


def print_clip_score_examples() -> None:
    """Print how output/image_path become a raw score and grade, using a fake scorer.

    Does not call the real CLIP model: that needs transformers,
    Pillow, and downloads ``openai/clip-vit-base-patch32`` (~600MB)
    on first use. Call ``score_clip_score(record)`` with no
    ``scorer=`` override to use the real model.
    """
    print("clip_score asks: how well does output's text align with the image?")
    print(f"  Real scorer: CLIP ({DEFAULT_MODEL}), cosine similarity * {RESCALE_FACTOR}.")
    print("  This demo uses a fake scorer to avoid an unprompted model download.")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("Matching caption and image.", {"output": "a dog on a beach", "image_path": "/data/dog_beach.jpg"}),
        ("Mismatched caption and image.", {"output": "a dog on a beach", "image_path": "/data/cat_couch.jpg"}),
        ("No image path on the record.", {"output": "a dog on a beach"}),
        ("Image path exists, but no output.", {"image_path": "/data/dog_beach.jpg"}),
        (
            "Case was cancelled (for example an audit stop).",
            {"output": "a dog on a beach", "image_path": "/data/dog_beach.jpg", "cancelled": True},
        ),
    )
    for story, record in examples:
        result = score_clip_score(record, scorer=_demo_scorer)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_clip_score_examples()
