from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


Box = tuple[float, float, float, float]
Key = tuple[str, int]


def parse_int_set(value: str | None) -> set[int] | None:
    if value is None or value.strip() == "":
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}

 
def load_gt(gt_dir: Path, *, cameras: set[str] | None = None) -> dict[Key, list[Box]]:
    """Load aligned 1920x1080 MOT-style GT files into xyxy boxes."""

    gt: dict[Key, list[Box]] = defaultdict(list)
    for path in sorted(gt_dir.glob("*_gt.txt")):
        camera_id = path.name.split("_", 1)[0]
        if cameras is not None and camera_id not in cameras:
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 6:
                    raise ValueError(f"Invalid GT row in {path}:{line_no}: {line}")
                frame_id = int(float(parts[0]))
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                if w <= 0.0 or h <= 0.0:
                    continue
                gt[(camera_id, frame_id)].append((x, y, x + w, y + h))
    return dict(gt)


def load_predictions(
    detections_path: Path,
    *,
    cameras: set[str] | None = None,
    class_ids: set[int] | None = None,
    score_thresh: float = 0.0,
    nms_iou_thresh: float | None = None,
) -> dict[Key, list[tuple[Box, float]]]:
    """Load detections.json and optionally apply eval-side filters/NMS."""

    data = json.loads(detections_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected detections JSON list, got {type(data).__name__}")

    predictions: dict[Key, list[tuple[Box, float]]] = defaultdict(list)
    for idx, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"Expected detection record dict at index {idx}")
        camera_id = str(record.get("camera_id", ""))
        if cameras is not None and camera_id not in cameras:
            continue
        frame_id = int(record["frame_id"])
        boxes = record.get("boxes", []) or []
        scores = record.get("scores", []) or []
        labels = record.get("labels", []) or []
        for det_idx, box in enumerate(boxes):
            score = float(scores[det_idx]) if det_idx < len(scores) else 1.0
            label = int(labels[det_idx]) if det_idx < len(labels) else -1
            if score < score_thresh:
                continue
            if class_ids is not None and label not in class_ids:
                continue
            if len(box) < 4:
                continue
            x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            if x2 <= x1 or y2 <= y1:
                continue
            predictions[(camera_id, frame_id)].append(((x1, y1, x2, y2), score, label))
    if nms_iou_thresh is not None:
        predictions = apply_nms(predictions, nms_iou_thresh)
    return {
        key: [(box, score) for box, score, _label in values]
        for key, values in predictions.items()
    }


def apply_nms(
    predictions: dict[Key, list[tuple[Box, float, int]]],
    iou_thresh: float,
) -> dict[Key, list[tuple[Box, float, int]]]:
    """Apply simple class-agnostic NMS independently per camera/frame."""

    filtered: dict[Key, list[tuple[Box, float, int]]] = {}
    for key, items in predictions.items():
        kept: list[tuple[Box, float, int]] = []
        for candidate in sorted(items, key=lambda item: item[1], reverse=True):
            if all(box_iou(candidate[0], kept_item[0]) < iou_thresh for kept_item in kept):
                kept.append(candidate)
        filtered[key] = kept
    return filtered


def box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def match_frame(gt_boxes: list[Box], pred_boxes: list[tuple[Box, float]], iou_thresh: float) -> tuple[int, int, int]:
    """Greedily match predictions to GT by descending score and IoU threshold."""

    matched_gt: set[int] = set()
    tp = 0
    fp = 0
    for pred_box, _score in sorted(pred_boxes, key=lambda item: item[1], reverse=True):
        best_iou = 0.0
        best_idx = -1
        for idx, gt_box in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            current_iou = box_iou(pred_box, gt_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_thresh:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def metric_summary(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate(
    gt: dict[Key, list[Box]],
    predictions: dict[Key, list[tuple[Box, float]]],
    *,
    iou_thresh: float,
    start_frame: int | None = None,
    end_frame: int | None = None,
    include_gt_only: bool = False,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Compute overall and per-camera TP/FP/FN, precision, recall, and F1."""

    pred_keys = set(predictions)
    gt_keys = set(gt)
    keys = pred_keys | gt_keys if include_gt_only or start_frame is not None or end_frame is not None else pred_keys

    if start_frame is not None:
        keys = {key for key in keys if key[1] >= start_frame}
    if end_frame is not None:
        keys = {key for key in keys if key[1] <= end_frame}

    total_tp = total_fp = total_fn = 0
    per_camera_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "frames": 0})
    for key in sorted(keys):
        camera_id, _frame_id = key
        frame_gt = gt.get(key, [])
        frame_pred = predictions.get(key, [])
        tp, fp, fn = match_frame(frame_gt, frame_pred, iou_thresh)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_camera_counts[camera_id]["tp"] += tp
        per_camera_counts[camera_id]["fp"] += fp
        per_camera_counts[camera_id]["fn"] += fn
        per_camera_counts[camera_id]["frames"] += 1

    overall = metric_summary(total_tp, total_fp, total_fn)
    overall.update(
        {
            "iou_thresh": iou_thresh,
            "frames": len(keys),
            "gt_boxes": sum(len(gt.get(key, [])) for key in keys),
            "pred_boxes": sum(len(predictions.get(key, [])) for key in keys),
        }
    )
    per_camera: dict[str, dict[str, Any]] = {}
    for camera_id, counts in sorted(per_camera_counts.items()):
        metrics = metric_summary(counts["tp"], counts["fp"], counts["fn"])
        metrics["frames"] = counts["frames"]
        per_camera[camera_id] = metrics
    return overall, per_camera


def write_csv(path: Path, overall: dict[str, Any], per_camera: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scope", "frames", "tp", "fp", "fn", "precision", "recall", "f1"])
        writer.writerow(
            [
                "overall",
                overall["frames"],
                overall["tp"],
                overall["fp"],
                overall["fn"],
                f"{overall['precision']:.6f}",
                f"{overall['recall']:.6f}",
                f"{overall['f1']:.6f}",
            ]
        )
        for camera_id, metrics in per_camera.items():
            writer.writerow(
                [
                    camera_id,
                    metrics["frames"],
                    metrics["tp"],
                    metrics["fp"],
                    metrics["fn"],
                    f"{metrics['precision']:.6f}",
                    f"{metrics['recall']:.6f}",
                    f"{metrics['f1']:.6f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RACE detections.json against aligned GT by IoU/F1.")
    parser.add_argument("--detections", required=True, help="Path to RACE output detections.json.")
    parser.add_argument("--gt_dir", default="dataset_preprocessing/aligned_gt_1920")
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--score_thresh", type=float, default=0.0)
    parser.add_argument(
        "--nms_iou_thresh",
        type=float,
        default=0.5,
        help="Class-agnostic NMS IoU threshold applied before matching. Use <=0 to disable.",
    )
    parser.add_argument("--class_ids", default=None, help="Optional comma-separated detector class ids to keep.")
    parser.add_argument("--cameras", default=None, help="Optional comma-separated camera ids, e.g. c001,c002.")
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument(
        "--include_gt_only",
        action="store_true",
        help="Also count GT-only frame/camera pairs as FN. Default evaluates only frame/camera pairs present in detections.",
    )
    parser.add_argument("--output_csv", default=None)
    args = parser.parse_args()

    cameras = {item.strip() for item in args.cameras.split(",") if item.strip()} if args.cameras else None
    class_ids = parse_int_set(args.class_ids)
    nms_iou_thresh = args.nms_iou_thresh if args.nms_iou_thresh > 0.0 else None
    gt = load_gt(Path(args.gt_dir), cameras=cameras)
    predictions = load_predictions(
        Path(args.detections),
        cameras=cameras,
        class_ids=class_ids,
        score_thresh=args.score_thresh,
        nms_iou_thresh=nms_iou_thresh,
    )
    overall, per_camera = evaluate(
        gt,
        predictions,
        iou_thresh=args.iou_thresh,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        include_gt_only=args.include_gt_only,
    )

    print(
        "[Eval Detections] "
        f"gt_dir={args.gt_dir} "
        f"detections={args.detections} "
        f"iou={args.iou_thresh:.2f} "
        f"nms_iou={nms_iou_thresh if nms_iou_thresh is not None else 'off'} "
        f"score_thresh={args.score_thresh:.2f} "
        "gt_scale=1.0"
    )
    print(
        "[Eval Detections] "
        f"frames={overall['frames']} "
        f"gt_boxes={overall['gt_boxes']} "
        f"pred_boxes={overall['pred_boxes']} "
        f"TP={overall['tp']} FP={overall['fp']} FN={overall['fn']} "
        f"precision={overall['precision']:.4f} "
        f"recall={overall['recall']:.4f} "
        f"F1={overall['f1']:.4f}"
    )
    for camera_id, metrics in per_camera.items():
        print(
            f"[Eval Detections:{camera_id}] "
            f"frames={metrics['frames']} "
            f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f}"
        )

    if args.output_csv:
        write_csv(Path(args.output_csv), overall, per_camera)
        print(f"[Eval Detections] CSV saved: {args.output_csv}")


if __name__ == "__main__":
    main()
