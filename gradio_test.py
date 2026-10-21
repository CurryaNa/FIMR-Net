#!/usr/bin/env python3
"""FIMR-Net research platform: gland detection and whole-slide analysis."""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
import threading
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if (PROJECT_ROOT / "ultralytics").is_dir():
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .global_dedup import deduplicate_by_slide, map_detection_to_global
except ImportError:
    from global_dedup import deduplicate_by_slide, map_detection_to_global


CLASS_NAMES = ("Normal", "Negative", "Positive")


def summarize(counts, weights, cutoff):
    """Apply the paper's confidence-weighted ratio; never invent zero evidence."""
    positive = float(weights["Positive"])
    negative = float(weights["Negative"])
    denominator = positive + negative
    ratio = positive / denominator if denominator else None
    return {
        "counts": {name: counts[name] for name in CLASS_NAMES},
        "positive_weight": positive,
        "negative_weight": negative,
        "weighted_positive_ratio": ratio,
        "cutoff": cutoff,
        "risk_group": (
            "INSUFFICIENT_EVIDENCE" if ratio is None
            else "HGAIM" if ratio < cutoff else "LGAIM"
        ),
    }


def count_predictions(result, names):
    counts, weights = Counter(), Counter()
    if result.boxes is not None:
        for box in result.boxes:
            name = names[int(box.cls.item())]
            counts[name] += 1
            weights[name] += float(box.conf.item())
    return counts, weights


def make_chart(counts):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(CLASS_NAMES, [counts[name] for name in CLASS_NAMES])
    ax.set_title("Gland detection counts")
    ax.set_ylabel("Count")
    fig.tight_layout()
    try:
        with io.BytesIO() as buffer:
            fig.savefig(buffer, format="png", dpi=120)
            buffer.seek(0)
            with Image.open(buffer) as image:
                return image.copy()
    finally:
        plt.close(fig)


class Platform:
    """Callbacks share a model, but each preprocessing run has isolated files."""

    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.lock = threading.Lock()
        self.temporary_runs = ExitStack()

    def close(self):
        self.temporary_runs.close()

    def prediction_kwargs(self, confidence):
        kwargs = {
            "conf": confidence, "imgsz": self.args.imgsz,
            "iou": self.args.iou, "max_det": self.args.max_det,
            "save": False, "verbose": False,
        }
        if self.args.device is not None:
            kwargs["device"] = self.args.device
        return kwargs

    def detect(self, image, confidence):
        if image is None:
            raise ValueError("请先上传腺体图像。")
        with self.lock:
            result = self.model.predict(
                image.convert("RGB"), **self.prediction_kwargs(confidence)
            )[0]
            counts, weights = count_predictions(result, self.model.names)
            annotated = Image.fromarray(result.plot()[..., ::-1])
            chart = make_chart(counts)
        report = summarize(counts, weights, self.args.patient_cutoff)
        # A single tile is not sufficient for a patient-level clinical conclusion.
        report.pop("risk_group")
        report.pop("cutoff")
        report["confidence_threshold"] = confidence
        report["note"] = "仅为当前图像的腺体统计，不代表患者级风险分层。"
        return annotated, chart, report

    def split_image(self, image):
        if image is None:
            raise ValueError("请先上传整图。")
        try:
            from .preprocess import process_slide
        except ImportError:
            from preprocess import process_slide

        directory = Path(self.temporary_runs.enter_context(
            tempfile.TemporaryDirectory(prefix="fimrnet-platform-")
        ))
        source = directory / "slide.png"
        image.convert("RGB").save(source)
        output = directory / "tiles"
        preprocess_args = argparse.Namespace(
            scale=1.5, tile_size=2048, overlap=0.25,
            mask_downsample=32, closing_kernel=5,
            tissue_threshold=0.8, std_threshold=15.0, overwrite=False,
        )
        rows = process_slide(source, output, preprocess_args)
        paths = [str(output / "slide" / row["tile_name"]) for row in rows if row["kept"]]
        metadata = {
            row["tile_name"]: {
                "slide_id": "slide",
                "tile_name": row["tile_name"],
                "x": int(row["x"]),
                "y": int(row["y"]),
                "width": int(row["original_tile_width"]),
                "height": int(row["original_tile_height"]),
            }
            for row in rows
            if row["kept"]
        }
        state = {"tiles": paths, "candidate_count": len(rows), "metadata": metadata}
        message = f"质控保留 {len(paths)}/{len(rows)} 张切片，预览最多显示 {self.args.preview_limit} 张。"
        if not paths:
            message += " 没有切片满足论文质控条件，无法进行风险分层。"
        return paths[:self.args.preview_limit], state, message

    def detect_all_tiles(self, state, confidence):
        if not state or not state.get("tiles"):
            return [], None, "请先上传整图并切片；无有效切片时无法分层。", {}
        detections, previews = [], []
        skipped_padding_centers = 0
        with self.lock:
            for result in self.model.predict(
                source=state["tiles"], stream=True, **self.prediction_kwargs(confidence)
            ):
                tile_name = Path(result.path).name
                metadata = state["metadata"][tile_name]
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        local_x1, local_y1, local_x2, local_y2 = map(float, box.xyxy[0].tolist())
                        row = map_detection_to_global(
                            {
                                "detection_id": f"D{len(detections) + 1:08d}",
                                "class_id": class_id,
                                "class_name": self.model.names[class_id],
                                "confidence": float(box.conf.item()),
                                "local_x1": local_x1,
                                "local_y1": local_y1,
                                "local_x2": local_x2,
                                "local_y2": local_y2,
                            },
                            metadata,
                        )
                        if row is None:
                            skipped_padding_centers += 1
                        else:
                            detections.append(row)
                if len(previews) < self.args.preview_limit:
                    previews.append(Image.fromarray(result.plot()[..., ::-1]))
        retained, duplicate_matches = deduplicate_by_slide(
            detections, self.args.dedup_iou, self.args.class_aware_dedup
        )
        counts, weights = Counter(), Counter()
        for row in retained:
            counts[row["class_name"]] += 1
            weights[row["class_name"]] += row["confidence"]
        chart = make_chart(counts)
        report = summarize(counts, weights, self.args.patient_cutoff)
        report.update({
            "tile_count": len(state["tiles"]),
            "confidence_threshold": confidence,
            "cross_tile_deduplication": True,
            "global_dedup_iou": self.args.dedup_iou,
            "class_aware_deduplication": self.args.class_aware_dedup,
            "raw_detections": len(detections),
            "unique_glands": len(retained),
            "duplicates_removed": len(duplicate_matches),
            "class_conflict_duplicates": sum(row["class_conflict"] for row in duplicate_matches),
            "detections_skipped_in_edge_padding": skipped_padding_centers,
            "research_use_only": True,
        })
        ratio = report["weighted_positive_ratio"]
        if ratio is None:
            message = "未检出 Positive/Negative 腺体，有效证据不足，无法分层。"
        else:
            label = "高风险组" if report["risk_group"] == "HGAIM" else "低风险组"
            message = f"加权阳性比例 {ratio:.6f}；cutoff={self.args.patient_cutoff} → {report['risk_group']}（{label}）"
        if abs(confidence - 0.4) > 1e-8:
            message += "\n注意：当前置信度偏离论文默认值 0.4，仅用于探索。"
        message += "\n仅供研究使用，不作为临床诊断依据。"
        return previews, chart, message, report


def build_app(platform):
    import gradio as gr

    with gr.Blocks(title="FIMR-Net 腺体检测与风险分层平台") as app:
        gr.Markdown("# FIMR-Net 腺体检测与风险分层平台\n仅供研究使用，不作为临床诊断依据。")
        gr.Markdown(
            "论文预处理：1.5×放大 · 2048×2048切片 · 25%重叠 · Otsu组织掩膜与闭运算 · "
            "组织覆盖率≥0.8 · 灰度标准差≥15 · 边缘白色补齐。\n\n"
            "患者评分采用置信度加权；低于固定阈值时归入HGAIM。整图分析会将检测框映射回全局坐标，"
            "并对重叠切片中的重复腺体进行置信度优先去重。"
        )
        with gr.Tabs():
            with gr.TabItem("单图检测（腺体级）"):
                image = gr.Image(type="pil", label="上传腺体图像")
                confidence = gr.Slider(0.01, 0.99, value=0.4, step=0.01, label="置信度（论文默认0.4）")
                button = gr.Button("检测腺体")
                annotated = gr.Image(label="检测结果")
                chart = gr.Image(label="腺体数量")
                report = gr.JSON(label="当前图像统计（不是患者风险结论）")
                button.click(platform.detect, [image, confidence], [annotated, chart, report])
            with gr.TabItem("整图分析（患者/切片级）"):
                state = gr.State(None)
                wsi = gr.Image(type="pil", label="上传整图（JPEG/PNG/TIFF等栅格图像）")
                split_button = gr.Button("1. 按论文参数切片与质控")
                tiles = gr.Gallery(label="有效切片预览", columns=4)
                status = gr.Textbox(label="切片状态")
                confidence = gr.Slider(0.01, 0.99, value=0.4, step=0.01, label="置信度（论文默认0.4）")
                detect_button = gr.Button("2. 检测全部有效切片并分层")
                gallery = gr.Gallery(label="检测结果预览", columns=4)
                chart = gr.Image(label="全部切片的腺体数量")
                conclusion = gr.Textbox(label="研究性风险分层结果")
                report = gr.JSON(label="加权评分与运行参数")
                def split_and_clear(image):
                    previews, new_state, message = platform.split_image(image)
                    return previews, new_state, message, [], None, "", {}

                split_button.click(
                    split_and_clear, [wsi],
                    [tiles, state, status, gallery, chart, conclusion, report],
                )
                detect_button.click(platform.detect_all_tiles, [state, confidence], [gallery, chart, conclusion, report])
                # A new upload invalidates the previous upload's tile state and results.
                wsi.change(
                    lambda: (None, [], "请重新切片。", [], None, "", {}),
                    [], [state, tiles, status, gallery, chart, conclusion, report],
                )
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=896)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--dedup-iou", type=float, default=0.5)
    parser.add_argument("--class-aware-dedup", action="store_true")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--patient-cutoff", type=float, default=0.4776)
    parser.add_argument("--preview-limit", type=int, default=50)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Explicitly enable a public Gradio link")
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"Weights do not exist: {args.weights}")
    if not all(0 < value < 1 for value in (args.patient_cutoff, args.iou, args.dedup_iou)) or args.preview_limit < 1:
        parser.error("Invalid cutoff, IoU, or preview limit")
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    if not set(CLASS_NAMES).issubset(set(model.names.values())):
        parser.error("Model must contain Normal, Negative, and Positive classes")
    platform = Platform(model, args)
    try:
        build_app(platform).queue().launch(
            server_name=args.host, server_port=args.port, share=args.share
        )
    finally:
        platform.close()


if __name__ == "__main__":
    main()
