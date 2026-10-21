#!/usr/bin/env python3
"""Preprocess FOLH1 IHC whole-slide images for FIMR-Net gland detection."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


Image.MAX_IMAGE_PIXELS = None
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing input WSIs/images")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for retained tiles and manifest")
    parser.add_argument("--scale", type=float, default=1.5, help="Image enlargement factor")
    parser.add_argument("--tile-size", type=int, default=2048)
    parser.add_argument("--overlap", type=float, default=0.25, help="Fractional overlap between adjacent tiles")
    parser.add_argument("--mask-downsample", type=int, default=32)
    parser.add_argument("--closing-kernel", type=int, default=5)
    parser.add_argument("--tissue-threshold", type=float, default=0.8)
    parser.add_argument("--std-threshold", type=float, default=15.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing tile images")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    if args.scale <= 0:
        raise ValueError("--scale must be positive")
    if args.tile_size <= 0 or args.mask_downsample <= 0:
        raise ValueError("--tile-size and --mask-downsample must be positive")
    if not 0 <= args.overlap < 1:
        raise ValueError("--overlap must be in [0, 1)")
    if not 0 <= args.tissue_threshold <= 1:
        raise ValueError("--tissue-threshold must be in [0, 1]")
    if args.closing_kernel <= 0:
        raise ValueError("--closing-kernel must be positive")


def build_tissue_mask(rgb: np.ndarray, downsample: int, closing_kernel: int) -> tuple[np.ndarray, float]:
    """Return a full-resolution Boolean tissue mask and the Otsu threshold."""
    height, width = rgb.shape[:2]
    low_width = max(1, round(width / downsample))
    low_height = max(1, round(height / downsample))
    low_rgb = cv2.resize(rgb, (low_width, low_height), interpolation=cv2.INTER_AREA)
    low_gray = cv2.cvtColor(low_rgb, cv2.COLOR_RGB2GRAY)

    # IHC tissue is generally darker than the near-white slide background.
    otsu_threshold, low_mask = cv2.threshold(
        low_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel = np.ones((closing_kernel, closing_kernel), dtype=np.uint8)
    low_mask = cv2.morphologyEx(low_mask, cv2.MORPH_CLOSE, kernel)
    full_mask = cv2.resize(low_mask, (width, height), interpolation=cv2.INTER_NEAREST) > 0
    return full_mask, float(otsu_threshold)


def pad_rgb(tile: np.ndarray, tile_size: int) -> np.ndarray:
    """Pad an edge tile to tile_size x tile_size using a white background."""
    height, width = tile.shape[:2]
    if height == tile_size and width == tile_size:
        return tile
    padded = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
    padded[:height, :width] = tile
    return padded


def pad_mask(mask: np.ndarray, tile_size: int) -> np.ndarray:
    height, width = mask.shape
    if height == tile_size and width == tile_size:
        return mask
    padded = np.zeros((tile_size, tile_size), dtype=bool)
    padded[:height, :width] = mask
    return padded


def process_slide(image_path: Path, output_dir: Path, args: argparse.Namespace) -> list[dict]:
    slide_id = image_path.stem
    slide_output = output_dir / slide_id
    slide_output.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        resized_width = max(1, round(image.width * args.scale))
        resized_height = max(1, round(image.height * args.scale))
        resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
        rgb = np.asarray(resized, dtype=np.uint8)

    tissue_mask, otsu_threshold = build_tissue_mask(
        rgb, args.mask_downsample, args.closing_kernel
    )
    height, width = rgb.shape[:2]
    stride = max(1, round(args.tile_size * (1 - args.overlap)))
    rows: list[dict] = []

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            raw_tile = rgb[y : min(y + args.tile_size, height), x : min(x + args.tile_size, width)]
            raw_mask = tissue_mask[
                y : min(y + args.tile_size, height), x : min(x + args.tile_size, width)
            ]
            tile = pad_rgb(raw_tile, args.tile_size)
            tile_mask = pad_mask(raw_mask, args.tile_size)
            gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
            tissue_ratio = float(tile_mask.mean())
            gray_std = float(gray.std(ddof=1))
            keep = tissue_ratio >= args.tissue_threshold and gray_std >= args.std_threshold
            tile_name = f"{slide_id}_{y}_{x}.png"

            if keep:
                tile_path = slide_output / tile_name
                if args.overwrite or not tile_path.exists():
                    Image.fromarray(tile).save(tile_path)

            rows.append(
                {
                    "slide_id": slide_id,
                    "source_image": image_path.name,
                    "tile_name": tile_name,
                    "x": x,
                    "y": y,
                    "original_tile_width": raw_tile.shape[1],
                    "original_tile_height": raw_tile.shape[0],
                    "tissue_ratio": f"{tissue_ratio:.6f}",
                    "gray_std": f"{gray_std:.6f}",
                    "otsu_threshold": f"{otsu_threshold:.2f}",
                    "kept": int(keep),
                }
            )

    del rgb, tissue_mask
    gc.collect()
    return rows


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        path for path in args.input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {args.input_dir}")

    config = {
        "paths_recorded": False,
        "scale": args.scale,
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "stride": max(1, round(args.tile_size * (1 - args.overlap))),
        "mask_downsample": args.mask_downsample,
        "closing_kernel": args.closing_kernel,
        "tissue_threshold": args.tissue_threshold,
        "std_threshold": args.std_threshold,
        "edge_policy": "white padding",
        "tissue_mask": "grayscale Otsu inverse threshold followed by morphological closing",
    }
    (args.output_dir / "preprocess_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    manifest_path = args.output_dir / "tiles_manifest.csv"
    fieldnames = [
        "slide_id",
        "source_image",
        "tile_name",
        "x",
        "y",
        "original_tile_width",
        "original_tile_height",
        "tissue_ratio",
        "gray_std",
        "otsu_threshold",
        "kept",
    ]
    total_candidates = total_kept = 0
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for image_path in tqdm(image_paths, desc="Preprocessing slides"):
            rows = process_slide(image_path, args.output_dir, args)
            writer.writerows(rows)
            file.flush()
            total_candidates += len(rows)
            total_kept += sum(row["kept"] for row in rows)
            print(f"{image_path.name}: kept {sum(row['kept'] for row in rows)}/{len(rows)} tiles")

    print(f"Finished: kept {total_kept}/{total_candidates} tiles")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
