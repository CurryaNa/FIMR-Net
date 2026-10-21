#!/usr/bin/env python3
"""Run gland detection and confidence-weighted patient risk stratification."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Prefer an adjacent source checkout when available.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if (PROJECT_ROOT / "ultralytics").is_dir():
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .global_dedup import (
        deduplicate_by_slide,
        load_manifest,
        load_manifest_slide_ids,
        map_detection_to_global,
        summarize_by_slide,
    )
except ImportError:
    from global_dedup import (
        deduplicate_by_slide,
        load_manifest,
        load_manifest_slide_ids,
        map_detection_to_global,
        summarize_by_slide,
    )


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
REQUIRED_CLASSES = {"Normal", "Negative", "Positive"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="Tile image or directory; directories are searched recursively")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/fimrnet_inference"))
    parser.add_argument("--conf", type=float, default=0.4, help="Paper gland-detection confidence threshold")
    parser.add_argument("--patient-cutoff", type=float, default=0.4776)
    parser.add_argument("--iou", type=float, default=0.7, help="Tile-level NMS IoU threshold")
    parser.add_argument("--dedup-iou", type=float, default=0.5, help="WSI-level cross-tile deduplication IoU")
    parser.add_argument("--manifest", type=Path, default=None, help="Preprocessing manifest; auto-detected under the source directory")
    parser.add_argument("--no-global-dedup", action="store_true", help="Disable cross-tile deduplication even when a manifest is available")
    parser.add_argument("--class-aware-dedup", action="store_true", help="Only merge duplicates with the same predicted class")
    parser.add_argument("--imgsz", type=int, default=896)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default=None)
    parser.add_argument("--patient-id-regex", default=None, help="Regex whose first capture group is the patient ID")
    parser.add_argument("--save-annotated", action="store_true")
    parser.add_argument("--data", type=Path, default=None, help="Optional dataset YAML for official detector evaluation")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights do not exist: {args.weights}")
    if not args.source.exists():
        raise FileNotFoundError(f"Source does not exist: {args.source}")
    if args.data is not None and not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {args.data}")
    if args.manifest is not None and not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {args.manifest}")
    if not all(0 < value < 1 for value in (args.conf, args.patient_cutoff, args.iou, args.dedup_iou)):
        raise ValueError("--conf, --patient-cutoff, --iou, and --dedup-iou must be between 0 and 1")
    if args.patient_id_regex is not None and re.compile(args.patient_id_regex).groups < 1:
        raise ValueError("--patient-id-regex must contain at least one capture group")


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.lower() in SUPPORTED_EXTENSIONS else []
    return sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def patient_id_from_path(path: Path, source: Path, pattern: re.Pattern | None) -> str:
    if pattern is not None:
        match = pattern.search(path.stem)
        if match is None:
            raise ValueError(f"Patient ID regex did not match: {path.name}")
        return match.group(1)
    if source.is_dir() and path.parent != source:
        return path.parent.name
    return path.stem.split("-", maxsplit=1)[0]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def official_detector_evaluation(model, args: argparse.Namespace) -> dict | None:
    if args.data is None:
        return None
    kwargs = {
        "data": str(args.data),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "verbose": False,
        "plots": True,
        "project": str(args.output_dir),
        "name": "detector_evaluation",
        "exist_ok": True,
    }
    if args.device is not None:
        kwargs["device"] = args.device
    result = model.val(**kwargs)
    metrics = {
        "split": args.split,
        "mAP50": float(result.box.map50),
        "mAP50_95": float(result.box.map),
        "mean_precision": float(result.box.mp),
        "mean_recall": float(result.box.mr),
    }
    precision, recall = metrics["mean_precision"], metrics["mean_recall"]
    metrics["f1_from_mean_precision_recall"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return metrics


def resolve_manifest(args: argparse.Namespace) -> Path | None:
    """Return an explicit or adjacent manifest unless deduplication is disabled."""
    if args.no_global_dedup:
        return None
    if args.manifest is not None:
        return args.manifest
    candidate = args.source / "tiles_manifest.csv" if args.source.is_dir() else None
    return candidate if candidate is not None and candidate.is_file() else None


def main() -> None:
    args = parse_args()
    validate_args(args)
    from ultralytics import YOLO

    image_paths = collect_images(args.source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(args.patient_id_regex) if args.patient_id_regex else None
    manifest_path = resolve_manifest(args)
    tile_metadata = load_manifest(manifest_path) if manifest_path is not None else None
    manifest_slide_ids = load_manifest_slide_ids(manifest_path) if manifest_path is not None else set()
    if not image_paths and manifest_path is None:
        raise FileNotFoundError(f"No supported images found under {args.source}")
    if tile_metadata is not None:
        images_by_key = {(path.parent.name, path.name): path for path in image_paths}
        missing_tiles = set(tile_metadata) - set(images_by_key)
        if missing_tiles:
            raise FileNotFoundError(
                f"Manifest references {len(missing_tiles)} retained tile(s) absent from the source directory"
            )
        image_paths = [images_by_key[key] for key in sorted(tile_metadata)]

    model = YOLO(str(args.weights))
    names = model.names
    available_classes = set(names.values())
    if not REQUIRED_CLASSES.issubset(available_classes):
        raise ValueError(
            f"Model classes must include {sorted(REQUIRED_CLASSES)}; found {sorted(available_classes)}"
        )

    predict_kwargs = {
        "source": [str(path) for path in image_paths],
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "max_det": args.max_det,
        "save": args.save_annotated,
        "project": str(args.output_dir),
        "name": "annotated",
        "exist_ok": True,
        "verbose": False,
        "stream": True,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device

    detection_rows: list[dict] = []
    tile_rows: list[dict] = []
    slide_tile_counts = Counter({slide_id: 0 for slide_id in manifest_slide_ids})
    skipped_padding_centers = 0
    prediction_results = model.predict(**predict_kwargs) if image_paths else []
    for result in prediction_results:
        image_path = Path(result.path)
        metadata = None
        if tile_metadata is not None:
            key = (image_path.parent.name, image_path.name)
            if key not in tile_metadata:
                raise RuntimeError(f"Prediction result is absent from the retained manifest: {image_path.name}")
            metadata = tile_metadata[key]
            patient_id = metadata["slide_id"]
        else:
            patient_id = patient_id_from_path(image_path, args.source, pattern)
        counts = Counter()
        weights = Counter()
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = names[class_id]
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                row = {
                    "detection_id": f"D{len(detection_rows) + 1:08d}",
                    "slide_id": patient_id,
                    "tile_name": image_path.name,
                    "tile_x": "",
                    "tile_y": "",
                    "tile_width": "",
                    "tile_height": "",
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "local_x1": x1,
                    "local_y1": y1,
                    "local_x2": x2,
                    "local_y2": y2,
                    "global_x1": "",
                    "global_y1": "",
                    "global_x2": "",
                    "global_y2": "",
                }
                if metadata is not None:
                    row = map_detection_to_global(row, metadata)
                    if row is None:
                        skipped_padding_centers += 1
                        continue
                counts[class_name] += 1
                weights[class_name] += confidence
                detection_rows.append(row)

        positive_weight = float(weights["Positive"])
        negative_weight = float(weights["Negative"])
        denominator = positive_weight + negative_weight
        ratio = positive_weight / denominator if denominator else None
        tile_rows.append(
            {
                "slide_id": patient_id,
                "tile_name": image_path.name,
                "normal_count": counts["Normal"],
                "negative_count": counts["Negative"],
                "positive_count": counts["Positive"],
                "negative_weight": f"{negative_weight:.8f}",
                "positive_weight": f"{positive_weight:.8f}",
                "weighted_positive_ratio": "" if ratio is None else f"{ratio:.8f}",
            }
        )
        slide_tile_counts[patient_id] += 1

    if tile_metadata is not None:
        unique_rows, duplicate_matches = deduplicate_by_slide(
            detection_rows, args.dedup_iou, args.class_aware_dedup
        )
    else:
        unique_rows, duplicate_matches = list(detection_rows), []

    gland_counts = Counter()
    glands: list[dict] = []
    for row in unique_rows:
        slide_id = row["slide_id"]
        gland_counts[slide_id] += 1
        glands.append({"gland_id": f"{slide_id}_G{gland_counts[slide_id]:06d}", **row})

    summaries = summarize_by_slide(unique_rows, dict(slide_tile_counts), args.patient_cutoff)
    patient_rows = [
        {
            "patient_id": row["slide_id"],
            **{key: ("" if value is None else value) for key, value in row.items() if key != "slide_id"},
        }
        for row in summaries
    ]

    detection_fields = [
        "detection_id", "slide_id", "tile_name", "tile_x", "tile_y", "tile_width", "tile_height",
        "class_id", "class_name", "confidence", "local_x1", "local_y1", "local_x2", "local_y2",
        "global_x1", "global_y1", "global_x2", "global_y2",
    ]

    write_csv(
        args.output_dir / "tile_predictions.csv",
        detection_rows,
        detection_fields,
    )
    write_csv(
        args.output_dir / "tile_summary.csv",
        tile_rows,
        ["slide_id", "tile_name", "normal_count", "negative_count", "positive_count", "negative_weight", "positive_weight", "weighted_positive_ratio"],
    )
    write_csv(args.output_dir / "unique_glands.csv", glands, ["gland_id", *detection_fields])
    write_csv(
        args.output_dir / "duplicate_matches.csv",
        duplicate_matches,
        [
            "slide_id", "retained_detection_id", "duplicate_detection_id", "retained_tile", "duplicate_tile",
            "retained_class", "duplicate_class", "retained_confidence", "duplicate_confidence", "global_iou", "class_conflict",
        ],
    )
    write_csv(
        args.output_dir / "patient_summary.csv",
        patient_rows,
        ["patient_id", "tile_count", "normal_count", "negative_count", "positive_count", "negative_weight", "positive_weight", "weighted_positive_ratio", "patient_cutoff", "risk_group"],
    )

    detector_metrics = official_detector_evaluation(model, args)
    config = {
        "paths_recorded": False,
        "images": len(image_paths),
        "patients": len(patient_rows),
        "confidence_threshold": args.conf,
        "patient_cutoff": args.patient_cutoff,
        "risk_rule": "weighted_positive_ratio < patient_cutoff -> HGAIM; otherwise LGAIM",
        "weighted_ratio_formula": "sum(Positive confidences) / (sum(Positive confidences) + sum(Negative confidences))",
        "zero_denominator_policy": "INSUFFICIENT_EVIDENCE",
        "imgsz": args.imgsz,
        "nms_iou": args.iou,
        "cross_tile_deduplication": tile_metadata is not None,
        "global_dedup_iou": args.dedup_iou if tile_metadata is not None else None,
        "class_aware_deduplication": args.class_aware_dedup if tile_metadata is not None else None,
        "deduplication_rule": (
            "confidence-descending greedy matching; same slide; different overlapping source tiles; "
            "global box IoU >= threshold; retain higher-confidence detection"
            if tile_metadata is not None
            else "not applied because no preprocessing manifest was supplied or discovered"
        ),
        "raw_detections": len(detection_rows),
        "unique_glands": len(glands),
        "duplicates_removed": len(duplicate_matches),
        "class_conflict_duplicates": sum(row["class_conflict"] for row in duplicate_matches),
        "detections_skipped_in_edge_padding": skipped_padding_centers,
        "detector_metrics": detector_metrics,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Processed {len(image_paths)} tiles from {len(patient_rows)} patients/slides")
    print("Results saved to the configured output directory.")


if __name__ == "__main__":
    main()
