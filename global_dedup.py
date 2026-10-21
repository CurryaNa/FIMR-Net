#!/usr/bin/env python3
"""WSI-coordinate mapping and cross-tile gland deduplication.

This module is intentionally independent of Ultralytics so the matching
behavior can be tested and reused by both command-line and web inference.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED_MANIFEST_COLUMNS = {
    "slide_id",
    "tile_name",
    "x",
    "y",
    "original_tile_width",
    "original_tile_height",
    "kept",
}


def _read_manifest_rows(path: Path) -> list[dict[str, str]]:
    """Read and validate a preprocessing manifest."""
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest is empty: {path}")
        missing = REQUIRED_MANIFEST_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        return list(reader)


def load_manifest(path: Path) -> dict[tuple[str, str], dict]:
    """Load retained tile geometry keyed by ``(slide_id, tile_name)``."""
    rows = _read_manifest_rows(path)

    metadata: dict[tuple[str, str], dict] = {}
    for row in rows:
        if int(row["kept"]) != 1:
            continue
        key = (row["slide_id"], row["tile_name"])
        if key in metadata:
            raise ValueError(f"Duplicate retained tile in manifest: {key}")
        metadata[key] = {
            "slide_id": row["slide_id"],
            "tile_name": row["tile_name"],
            "x": int(row["x"]),
            "y": int(row["y"]),
            "width": int(row["original_tile_width"]),
            "height": int(row["original_tile_height"]),
        }
    return metadata


def load_manifest_slide_ids(path: Path) -> set[str]:
    """Return every slide ID, including slides with no retained tiles."""
    return {row["slide_id"] for row in _read_manifest_rows(path)}


def map_detection_to_global(detection: dict, metadata: dict) -> dict | None:
    """Map a local box to WSI coordinates, excluding white-padding centers."""
    center_x = (float(detection["local_x1"]) + float(detection["local_x2"])) / 2
    center_y = (float(detection["local_y1"]) + float(detection["local_y2"])) / 2
    if not (0 <= center_x < metadata["width"] and 0 <= center_y < metadata["height"]):
        return None

    mapped = dict(detection)
    mapped.update(
        {
            "slide_id": metadata["slide_id"],
            "tile_name": metadata["tile_name"],
            "tile_x": metadata["x"],
            "tile_y": metadata["y"],
            "tile_width": metadata["width"],
            "tile_height": metadata["height"],
            "global_x1": metadata["x"] + float(detection["local_x1"]),
            "global_y1": metadata["y"] + float(detection["local_y1"]),
            "global_x2": metadata["x"] + float(detection["local_x2"]),
            "global_y2": metadata["y"] + float(detection["local_y2"]),
        }
    )
    return mapped


def boxes_iou(first: dict, second: dict) -> float:
    """Return IoU between two boxes expressed in WSI coordinates."""
    x1 = max(first["global_x1"], second["global_x1"])
    y1 = max(first["global_y1"], second["global_y1"])
    x2 = min(first["global_x2"], second["global_x2"])
    y2 = min(first["global_y2"], second["global_y2"])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    first_area = max(0.0, first["global_x2"] - first["global_x1"]) * max(
        0.0, first["global_y2"] - first["global_y1"]
    )
    second_area = max(0.0, second["global_x2"] - second["global_x1"]) * max(
        0.0, second["global_y2"] - second["global_y1"]
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def source_tiles_overlap(first: dict, second: dict) -> bool:
    """Return whether detections came from distinct, overlapping valid tiles."""
    if first["tile_name"] == second["tile_name"]:
        return False
    return (
        max(first["tile_x"], second["tile_x"])
        < min(first["tile_x"] + first["tile_width"], second["tile_x"] + second["tile_width"])
        and max(first["tile_y"], second["tile_y"])
        < min(first["tile_y"] + first["tile_height"], second["tile_y"] + second["tile_height"])
    )


def _deduplicate_slide(
    detections: list[dict], iou_threshold: float, class_aware: bool
) -> tuple[list[dict], list[dict]]:
    ordered = sorted(detections, key=lambda row: (-row["confidence"], row["detection_id"]))
    retained: list[dict] = []
    matches: list[dict] = []
    for candidate in ordered:
        best_match = None
        best_iou = 0.0
        for kept in retained:
            if class_aware and candidate["class_id"] != kept["class_id"]:
                continue
            if not source_tiles_overlap(candidate, kept):
                continue
            iou = boxes_iou(candidate, kept)
            if iou >= iou_threshold and iou > best_iou:
                best_match, best_iou = kept, iou
        if best_match is None:
            retained.append(candidate)
            continue
        matches.append(
            {
                "slide_id": candidate["slide_id"],
                "retained_detection_id": best_match["detection_id"],
                "duplicate_detection_id": candidate["detection_id"],
                "retained_tile": best_match["tile_name"],
                "duplicate_tile": candidate["tile_name"],
                "retained_class": best_match["class_name"],
                "duplicate_class": candidate["class_name"],
                "retained_confidence": best_match["confidence"],
                "duplicate_confidence": candidate["confidence"],
                "global_iou": best_iou,
                "class_conflict": int(best_match["class_id"] != candidate["class_id"]),
            }
        )
    return retained, matches


def deduplicate_by_slide(
    detections: list[dict], iou_threshold: float = 0.5, class_aware: bool = False
) -> tuple[list[dict], list[dict]]:
    """Deduplicate detections independently within each slide."""
    if not 0 < iou_threshold < 1:
        raise ValueError("iou_threshold must be between 0 and 1")
    by_slide: dict[str, list[dict]] = defaultdict(list)
    for row in detections:
        by_slide[row["slide_id"]].append(row)

    retained: list[dict] = []
    matches: list[dict] = []
    for slide_id in sorted(by_slide):
        slide_retained, slide_matches = _deduplicate_slide(
            by_slide[slide_id], iou_threshold, class_aware
        )
        retained.extend(slide_retained)
        matches.extend(slide_matches)
    return retained, matches


def summarize_by_slide(
    detections: list[dict], slide_tile_counts: dict[str, int], cutoff: float
) -> list[dict]:
    """Build confidence-weighted patient/slide summaries from retained glands."""
    by_slide: dict[str, list[dict]] = defaultdict(list)
    for row in detections:
        by_slide[row["slide_id"]].append(row)

    summaries: list[dict] = []
    for slide_id in sorted(set(by_slide) | set(slide_tile_counts)):
        rows = by_slide[slide_id]
        counts = Counter(row["class_name"] for row in rows)
        weights = Counter()
        for row in rows:
            weights[row["class_name"]] += float(row["confidence"])
        positive_weight = float(weights["Positive"])
        negative_weight = float(weights["Negative"])
        denominator = positive_weight + negative_weight
        ratio = positive_weight / denominator if denominator else None
        summaries.append(
            {
                "slide_id": slide_id,
                "tile_count": int(slide_tile_counts.get(slide_id, 0)),
                "normal_count": counts["Normal"],
                "negative_count": counts["Negative"],
                "positive_count": counts["Positive"],
                "negative_weight": negative_weight,
                "positive_weight": positive_weight,
                "weighted_positive_ratio": ratio,
                "patient_cutoff": cutoff,
                "risk_group": (
                    "INSUFFICIENT_EVIDENCE"
                    if ratio is None
                    else "HGAIM"
                    if ratio < cutoff
                    else "LGAIM"
                ),
            }
        )
    return summaries
