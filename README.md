# FIMR-Net code release (work in progress)

This directory contains the anonymous research implementation of FIMR-Net.
No author names, affiliations, contact information, or private dataset links
are included.

The current release covers four stages:

1. WSI/image enlargement, overlapping tiling, and quality filtering.
2. Three-class gland detector training (`Normal`, `Negative`, `Positive`).
3. WSI-coordinate cross-tile deduplication of overlapping gland detections.
4. Confidence-weighted patient scoring and risk stratification.

> **Research use only.** This software is not a medical device and must not be
> used as the sole basis for clinical diagnosis or treatment decisions.

## 1. Preprocess whole-slide images

The defaults match the paper: 1.5x enlargement, 2048-pixel tiles, 25% overlap,
low-resolution Otsu tissue masking, tissue coverage >= 0.8, and grayscale
standard deviation >= 15. Edge tiles are padded with white pixels.

```bash
python fimrnet_release/preprocess.py \
  --input-dir /path/to/wsis \
  --output-dir /path/to/tiles
```

Outputs are arranged by slide ID. `tiles_manifest.csv` records the coordinates
and quality measurements of every candidate tile.

## 2. Train the gland detector

The training entry point accepts a model configuration and dataset YAML.

```bash
python fimrnet_release/train.py \
  --model-config /path/to/model.yaml \
  --pretrained /path/to/pretrained.pt \
  --data /path/to/dataset.yaml \
  --device 0
```

The default training settings are: image size 896,
150 epochs, batch size 16, SGD, AMP disabled, and mosaic disabled for the final
10 epochs.

The paper's final detector is described as YOLOv12 + Focal Loss. This training
wrapper can accept the corresponding model YAML and weights, but the exact
Focal Loss implementation used to train the released checkpoint must be
verified and included before claiming full training reproducibility.

## 3. Infer and stratify patients

The default gland confidence threshold is 0.4 and the fixed patient cutoff is
0.4776. A patient is classified as HGAIM when its confidence-weighted positive
gland ratio is below the cutoff.

```bash
python fimrnet_release/infer.py \
  --weights /path/to/best.pt \
  --source /path/to/preprocessed_tiles \
  --output-dir runs/fimrnet_inference \
  --device 0 \
  --save-annotated
```

When `--source` is a directory created by `preprocess.py`,
`tiles_manifest.csv` is discovered automatically and WSI-level global
deduplication is enabled. A manifest elsewhere can be supplied explicitly:

```bash
python fimrnet_release/infer.py \
  --weights /path/to/best.pt \
  --source /path/to/preprocessed_tiles \
  --manifest /path/to/tiles_manifest.csv \
  --dedup-iou 0.5 \
  --output-dir runs/fimrnet_inference
```

After conventional NMS within each tile, local boxes are mapped to the 1.5x
enlarged-WSI coordinate system using the manifest. Detections are compared
only when they belong to the same slide and originate from different,
spatially overlapping source tiles. Confidence-descending greedy matching
defines boxes with global IoU >= 0.5 as duplicates and retains the
higher-confidence box. Matching is class-agnostic by default; a conflicting
class assignment is retained from the higher-confidence box and recorded in
the audit table. Use `--class-aware-dedup` to restrict matching to equal
classes, or `--no-global-dedup` for an explicit tile-aggregation baseline.

Detections whose centers fall in white edge padding are excluded before
deduplication. If no manifest is supplied or discovered, global coordinates
cannot be recovered, so the CLI reports that deduplication was not applied
and falls back to tile-wise aggregation.

A slide recorded in the manifest but left with no retained tile or no
Positive/Negative gland is still written to `patient_summary.csv` as
`INSUFFICIENT_EVIDENCE`; it is never silently removed from the cohort.

The command writes:

- `tile_predictions.csv`: one row per raw tile-level detection, including
  local and (when available) global coordinates.
- `unique_glands.csv`: retained gland detections after global deduplication.
- `duplicate_matches.csv`: auditable retained/removed pairs, global IoU, and
  class-conflict flags.
- `tile_summary.csv`: raw counts and weighted scores for each tile.
- `patient_summary.csv`: patient/slide-level ratio and risk group calculated
  from `unique_glands.csv` when deduplication is active.
- `run_config.json`: the exact inference parameters.
- `annotated/`: optional visual results.

If labeled test data are available, official Ultralytics detector metrics can
also be calculated:

```bash
python fimrnet_release/infer.py \
  --weights /path/to/best.pt \
  --source /path/to/test/images \
  --data /path/to/data.yaml \
  --split test \
  --output-dir runs/fimrnet_test
```

## 4. Launch the web platform

The platform provides gland-level and whole-slide tabs. It reuses
`preprocess.py`, maps whole-slide detections to global coordinates, applies
the same cross-tile deduplication as `infer.py`, and uses the same default
confidence (0.4), global IoU (0.5), and patient cutoff (0.4776).

```bash
python fimrnet_release/gradio_test.py \
  --weights /path/to/best.pt \
  --device 0
```

Open `http://127.0.0.1:7860`. Public sharing is disabled unless `--share` is
explicitly supplied. Do not upload identifiable clinical data to public links.
The UI is compatible with Gradio 3.35.x. Each preprocessing run uses isolated
temporary files, cleaned up when the platform exits. Previews are capped at 50
tiles by default, but scoring includes all retained tiles. The whole-slide tab
analyzes one uploaded slide, not multiple slides belonging to one patient.

## Patient IDs

For output created by `preprocess.py`, the parent directory name is used as the
patient/slide ID. For a flat input directory, the default fallback takes the
part of the filename before the first hyphen. A custom regular expression can
be supplied with `--patient-id-regex`; its first capture group is used.

## Global deduplication scope and limitations

The release implements the experimentally evaluated confidence-ordered,
class-agnostic global-IoU method. It is deterministic for a fixed set of
tile-level predictions. It does not fuse box coordinates or average class
probabilities: the highest-confidence detection represents each matched
gland. The method also assumes that the preprocessing manifest and tile files
remain paired; missing retained tiles cause a clear error rather than a
partial analysis.

The default global IoU of 0.5 is the primary setting. Alternative values
should be reported as sensitivity analyses rather than selected from the test
set. Conventional detector NMS remains a separate, within-tile operation
controlled by `--iou`.

## Anonymity and data privacy

The release contains code only, without patient data or model checkpoints.
Example commands use generic paths and do not identify private experiment
directories. Preprocessing manifests store only input basenames, and custom
JSON run summaries omit input/output/weight paths.

This does **not** anonymize input images or filenames. Before sharing data or
outputs, replace patient/slide IDs and filenames with approved anonymous IDs
and remove identifying image content. Keep any ID mapping outside the release.
The inference grouping logic uses the IDs supplied by the input filenames or
directories; it does not silently rename or change patient group membership.

Third-party training/validation tools may still write local paths into their
own logs or checkpoint metadata. Review these artifacts separately before
publication. Anonymous review submissions may also need to omit identifying
paper titles or links, depending on the venue's requirements.
