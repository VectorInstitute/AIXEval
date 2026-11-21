"""grounding_accuracy raw score: IoU-thresholded region match, higher-is-better.

Literature: Intersection-over-Union (IoU) region matching, the
standard visual-grounding evaluation convention (e.g. RefCOCO,
RefCOCO+). A 0.5 IoU threshold for binary "correct region" is the
same convention used throughout detection and grounding benchmarks
(PASCAL VOC, COCO, RefCOCO-style accuracy).

Pure geometry, no image I/O: this module compares two declared
bounding boxes (``predicted_bbox``, ``gold_bbox``, each ``[x1, y1, x2,
y2]``); it never loads the image itself. Not the same construct as
``grounding`` (text-faithfulness-to-evidence); this is
visual-region-pointing accuracy entirely, despite the similar name.
"""

from __future__ import annotations

from typing import Any

from aixeval.scorecard.na_reasons import require_na_reason
from aixeval.scorecard.profile import apply_grade, label_for_band


METRIC_ID = "grounding_accuracy"
# Declared convention, not a free parameter: IoU >= 0.5 counts as a
# correct region match, matching PASCAL VOC / RefCOCO-style accuracy.
IOU_CORRECT_THRESHOLD = 0.5


def score_grounding_accuracy(record: dict[str, Any]) -> dict[str, Any]:
    """Score whether ``predicted_bbox`` matches ``gold_bbox`` above the IoU threshold.

    Reads ``predicted_bbox`` and ``gold_bbox``, each a ``[x1, y1, x2,
    y2]`` list of 4 floats. Raw score is 1.0 if IoU >= 0.5, else 0.0.
    Higher is better. Missing or invalid boxes are N/A, never a
    silent 0.0.
    """
    if record.get("cancelled") is True or record.get("evidence_status") == "cancelled":
        return _na("cancelled")

    predicted, predicted_ok = _optional_bbox(record, "predicted_bbox")
    gold, gold_ok = _optional_bbox(record, "gold_bbox")
    if not predicted_ok or not gold_ok:
        return _na("missing_evidence")
    if gold is None:
        return _na("no_gold")
    if predicted is None:
        return _na("missing_output")

    iou = _iou(predicted, gold)
    return _scored(1.0 if iou >= IOU_CORRECT_THRESHOLD else 0.0)


def _iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def _optional_bbox(record: dict[str, Any], key: str) -> tuple[list[float] | None, bool]:
    """Return a 4-float bbox and whether the field type is usable."""
    if key not in record or record[key] is None:
        return None, True
    value = record[key]
    if not isinstance(value, list) or len(value) != 4:
        return None, False
    if not all(isinstance(coordinate, int | float) and not isinstance(coordinate, bool) for coordinate in value):
        return None, False
    return [float(coordinate) for coordinate in value], True


def _scored(raw_score: float) -> dict[str, Any]:
    return apply_grade({"metric_id": METRIC_ID, "status": "scored", "raw_score": raw_score})


def _na(reason: str) -> dict[str, Any]:
    return {"metric_id": METRIC_ID, "status": "na", "na_reason": require_na_reason(reason)}


def print_grounding_accuracy_examples() -> None:
    """Print how predicted/gold boxes become a raw score and grade."""
    print("grounding_accuracy asks: does predicted_bbox point to the correct region?")
    print(f"  IoU(predicted_bbox, gold_bbox) >= {IOU_CORRECT_THRESHOLD} -> raw_score 1.0 (grade 4).")
    print("  Otherwise -> raw_score 0.0 (grade 0). Higher is better.")
    print()
    examples: tuple[tuple[str, dict[str, Any]], ...] = (
        ("Exact box match.", {"predicted_bbox": [10, 10, 50, 50], "gold_bbox": [10, 10, 50, 50]}),
        (
            "Mostly overlapping box (IoU above threshold).",
            {"predicted_bbox": [12, 12, 52, 52], "gold_bbox": [10, 10, 50, 50]},
        ),
        (
            "Barely overlapping box (IoU below threshold).",
            {"predicted_bbox": [45, 45, 85, 85], "gold_bbox": [10, 10, 50, 50]},
        ),
        ("No gold box on the record.", {"predicted_bbox": [10, 10, 50, 50]}),
        (
            "Case was cancelled (for example an audit stop).",
            {"predicted_bbox": [10, 10, 50, 50], "gold_bbox": [10, 10, 50, 50], "cancelled": True},
        ),
    )
    for story, record in examples:
        result = score_grounding_accuracy(record)
        print(story)
        print(f"  record: {record}")
        print(f"  result: {result}")
        if result.get("status") == "scored":
            grade = int(result["grade"])
            print(f"  grade:  {grade} ({label_for_band(grade)})")
        print()


if __name__ == "__main__":
    print_grounding_accuracy_examples()
