#!/usr/bin/env python3
"""Train the three-class FIMR-Net gland detector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Prefer an adjacent source checkout when available.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if (PROJECT_ROOT / "ultralytics").is_dir():
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True, help="Ultralytics model YAML or checkpoint")
    parser.add_argument("--pretrained", type=Path, default=None, help="Optional pretrained checkpoint loaded into the model")
    parser.add_argument("--data", type=Path, required=True, help="YOLO dataset YAML")
    parser.add_argument("--imgsz", type=int, default=896)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0, 0,1, or cpu")
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--project", type=Path, default=Path("runs/fimrnet_train"))
    parser.add_argument("--name", default="yolo11_gland_detector")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_config.exists():
        raise FileNotFoundError(f"Model config/checkpoint does not exist: {args.model_config}")
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset YAML does not exist: {args.data}")
    if args.pretrained is not None and not args.pretrained.exists():
        raise FileNotFoundError(f"Pretrained checkpoint does not exist: {args.pretrained}")

    model = YOLO(str(args.model_config))
    if args.pretrained is not None:
        model.load(str(args.pretrained))

    train_kwargs = {
        "data": str(args.data),
        "cache": args.cache,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "single_cls": False,
        "batch": args.batch,
        "close_mosaic": args.close_mosaic,
        "workers": args.workers,
        "optimizer": args.optimizer,
        "amp": args.amp,
        "project": str(args.project),
        "name": args.name,
        "seed": args.seed,
        "deterministic": True,
        "exist_ok": args.exist_ok,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
