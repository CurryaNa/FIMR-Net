from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from fimrnet_release.global_dedup import (
    deduplicate_by_slide,
    load_manifest,
    load_manifest_slide_ids,
    map_detection_to_global,
    summarize_by_slide,
)


class GlobalDeduplicationTests(unittest.TestCase):
    def test_maps_local_box_and_rejects_padding_center(self):
        metadata = {
            "slide_id": "demo_slide",
            "tile_name": "demo_slide_0_100.png",
            "x": 100,
            "y": 0,
            "width": 80,
            "height": 100,
        }
        detection = {
            "detection_id": "D1",
            "class_id": 2,
            "class_name": "Positive",
            "confidence": 0.9,
            "local_x1": 10.0,
            "local_y1": 20.0,
            "local_x2": 30.0,
            "local_y2": 40.0,
        }

        mapped = map_detection_to_global(detection, metadata)
        self.assertEqual(
            (mapped["global_x1"], mapped["global_y1"], mapped["global_x2"], mapped["global_y2"]),
            (110.0, 20.0, 130.0, 40.0),
        )

        padding_detection = dict(detection, local_x1=80.0, local_x2=90.0)
        self.assertIsNone(map_detection_to_global(padding_detection, metadata))

    def test_deduplicates_across_overlapping_tiles_and_audits_class_conflict(self):
        common = {
            "slide_id": "demo_slide",
            "tile_width": 100,
            "tile_height": 100,
            "global_x1": 60.0,
            "global_y1": 10.0,
            "global_x2": 90.0,
            "global_y2": 40.0,
        }
        high = dict(
            common,
            detection_id="D1",
            tile_name="demo_slide_0_0.png",
            tile_x=0,
            tile_y=0,
            class_id=2,
            class_name="Positive",
            confidence=0.9,
        )
        low = dict(
            common,
            detection_id="D2",
            tile_name="demo_slide_0_50.png",
            tile_x=50,
            tile_y=0,
            class_id=1,
            class_name="Negative",
            confidence=0.7,
        )

        retained, matches = deduplicate_by_slide([low, high], iou_threshold=0.5)

        self.assertEqual([row["detection_id"] for row in retained], ["D1"])
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["retained_detection_id"], "D1")
        self.assertEqual(matches[0]["duplicate_detection_id"], "D2")
        self.assertEqual(matches[0]["class_conflict"], 1)

    def test_preserves_same_tile_and_nonoverlapping_tile_detections(self):
        base = {
            "slide_id": "demo_slide",
            "tile_width": 100,
            "tile_height": 100,
            "class_id": 2,
            "class_name": "Positive",
            "confidence": 0.9,
            "global_x1": 10.0,
            "global_y1": 10.0,
            "global_x2": 30.0,
            "global_y2": 30.0,
        }
        rows = [
            dict(base, detection_id="D1", tile_name="a.png", tile_x=0, tile_y=0),
            dict(base, detection_id="D2", tile_name="a.png", tile_x=0, tile_y=0, confidence=0.8),
            dict(base, detection_id="D3", tile_name="b.png", tile_x=200, tile_y=0, confidence=0.7),
        ]

        retained, matches = deduplicate_by_slide(rows, iou_threshold=0.5)

        self.assertEqual({row["detection_id"] for row in retained}, {"D1", "D2", "D3"})
        self.assertEqual(matches, [])

    def test_class_aware_mode_preserves_conflicting_classes(self):
        base = {
            "slide_id": "demo_slide",
            "tile_width": 100,
            "tile_height": 100,
            "global_x1": 60.0,
            "global_y1": 10.0,
            "global_x2": 90.0,
            "global_y2": 40.0,
        }
        rows = [
            dict(base, detection_id="D1", tile_name="a.png", tile_x=0, tile_y=0, class_id=2, class_name="Positive", confidence=0.9),
            dict(base, detection_id="D2", tile_name="b.png", tile_x=50, tile_y=0, class_id=1, class_name="Negative", confidence=0.8),
        ]

        retained, matches = deduplicate_by_slide(rows, iou_threshold=0.5, class_aware=True)

        self.assertEqual(len(retained), 2)
        self.assertEqual(matches, [])

    def test_manifest_keeps_only_retained_tiles_and_rejects_missing_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            good = Path(directory) / "tiles_manifest.csv"
            with good.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["slide_id", "tile_name", "x", "y", "original_tile_width", "original_tile_height", "kept"],
                )
                writer.writeheader()
                writer.writerow({"slide_id": "demo", "tile_name": "kept.png", "x": 0, "y": 0, "original_tile_width": 100, "original_tile_height": 90, "kept": 1})
                writer.writerow({"slide_id": "demo", "tile_name": "dropped.png", "x": 50, "y": 0, "original_tile_width": 100, "original_tile_height": 90, "kept": 0})
            metadata = load_manifest(good)
            self.assertEqual(set(metadata), {("demo", "kept.png")})
            self.assertEqual(metadata[("demo", "kept.png")]["height"], 90)
            self.assertEqual(load_manifest_slide_ids(good), {"demo"})

            bad = Path(directory) / "bad.csv"
            bad.write_text("slide_id,tile_name\ndemo,tile.png\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing columns"):
                load_manifest(bad)

    def test_manifest_preserves_slide_with_no_retained_tiles(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "tiles_manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["slide_id", "tile_name", "x", "y", "original_tile_width", "original_tile_height", "kept"],
                )
                writer.writeheader()
                writer.writerow({"slide_id": "empty_slide", "tile_name": "dropped.png", "x": 0, "y": 0, "original_tile_width": 100, "original_tile_height": 100, "kept": 0})

            self.assertEqual(load_manifest(manifest), {})
            self.assertEqual(load_manifest_slide_ids(manifest), {"empty_slide"})

    def test_patient_summary_uses_retained_detections(self):
        detections = [
            {"slide_id": "demo", "class_name": "Normal", "confidence": 0.8},
            {"slide_id": "demo", "class_name": "Positive", "confidence": 0.6},
            {"slide_id": "demo", "class_name": "Negative", "confidence": 0.9},
        ]

        rows = summarize_by_slide(detections, {"demo": 4}, cutoff=0.4776)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["tile_count"], 4)
        self.assertEqual(rows[0]["normal_count"], 1)
        self.assertEqual(rows[0]["positive_count"], 1)
        self.assertEqual(rows[0]["negative_count"], 1)
        self.assertAlmostEqual(rows[0]["weighted_positive_ratio"], 0.4)
        self.assertEqual(rows[0]["risk_group"], "HGAIM")

        empty_rows = summarize_by_slide([], {"empty_slide": 0}, cutoff=0.4776)
        self.assertEqual(len(empty_rows), 1)
        self.assertIsNone(empty_rows[0]["weighted_positive_ratio"])
        self.assertEqual(empty_rows[0]["risk_group"], "INSUFFICIENT_EVIDENCE")


if __name__ == "__main__":
    unittest.main()
