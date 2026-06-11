from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pycuda.driver as cuda
import torch

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from RACE.sr_model.sr_batch_infer import SRBatchInfer


def parse_batch_sizes(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("batch sizes must be positive integers, e.g. 1,2,4,8")
    return values


def parse_engine_map(raw: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError(
                "engine_map must look like 1:path/to/b1.engine,2:path/to/b2.engine"
            )
        key, value = item.split(":", 1)
        batch_size = int(key.strip())
        engine_path = value.strip()
        if batch_size <= 0 or not engine_path:
            raise argparse.ArgumentTypeError(
                "engine_map entries must use a positive batch size and non-empty path"
            )
        mapping[batch_size] = engine_path
    if not mapping:
        raise argparse.ArgumentTypeError("engine_map cannot be empty")
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SR latency versus batch size K.")
    parser.add_argument("--video_dir", default="./dataset_preprocessing/aligned_videos_640")
    parser.add_argument("--sr_model", default=None, help="Single engine path used for all batch sizes.")
    parser.add_argument(
        "--engine_map",
        type=parse_engine_map,
        default=None,
        help="Per-K engine map, e.g. 1:temp_model/EDSR_x3_b1.engine,2:temp_model/EDSR_x3_b2.engine,4:temp_model/EDSR_x3_b4.engine",
    )
    parser.add_argument("--batch_sizes", type=parse_batch_sizes, default=parse_batch_sizes("1,2,4,8"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--num_source_frames", type=int, default=128)
    parser.add_argument("--input_width", type=int, default=640)
    parser.add_argument("--input_height", type=int, default=360)
    parser.add_argument("--output_csv", default="RACE/output_runtime/sr_latency.csv")
    parser.add_argument("--output_json", default="RACE/output_runtime/sr_latency.json")
    args = parser.parse_args()
    if args.input_width <= 0 or args.input_height <= 0:
        parser.error("--input_width and --input_height must be positive")
    if args.sr_model is None and args.engine_map is None:
        parser.error("one of --sr_model or --engine_map is required")
    return args


def discover_videos(video_dir: str) -> list[Path]:
    root = Path(video_dir)
    videos = sorted(root.glob("*_aligned.avi"))
    if not videos:
        videos = sorted(root.glob("*.avi"))
    if not videos:
        raise FileNotFoundError(f"No .avi videos found under {video_dir}")
    return videos


def load_video_frames(video_dir: str, limit: int, input_width: int, input_height: int) -> list[np.ndarray]:
    videos = discover_videos(video_dir)
    caps = [cv2.VideoCapture(str(path)) for path in videos]
    try:
        for path, cap in zip(videos, caps):
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {path}")

        frames: list[np.ndarray] = []
        while len(frames) < limit:
            progressed = False
            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    continue
                progressed = True
                if frame.shape[1] != input_width or frame.shape[0] != input_height:
                    frame = cv2.resize(frame, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
                frames.append(frame)
                if len(frames) >= limit:
                    break
            if not progressed:
                break

        if not frames:
            raise RuntimeError(f"Could not read frames from {video_dir}")
        return frames
    finally:
        for cap in caps:
            cap.release()


def make_batch(frames: list[np.ndarray], batch_size: int, offset: int) -> torch.Tensor:
    batch_frames = [frames[(offset + idx) % len(frames)] for idx in range(batch_size)]
    rgb = np.stack(
        [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) for frame in batch_frames],
        axis=0,
    )
    return torch.from_numpy(rgb).permute(0, 3, 1, 2).cuda().float().contiguous()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, pct))


def benchmark_batch_size(
    sr_model: SRBatchInfer,
    frames: list[np.ndarray],
    batch_size: int,
    warmup: int,
    repeat: int,
) -> dict[str, float | int]:
    batch = make_batch(frames, batch_size, offset=0)

    for _ in range(warmup):
        _ = sr_model.inference(batch)
    torch.cuda.synchronize()

    latencies_ms: list[float] = []
    for idx in range(repeat):
        batch = make_batch(frames, batch_size, offset=idx * batch_size)
        torch.cuda.synchronize()
        started = time.perf_counter()
        _ = sr_model.inference(batch)
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - started) * 1000.0)

    mean_ms = statistics.mean(latencies_ms)
    return {
        "batch_size": batch_size,
        "input_width": int(frames[0].shape[1]) if frames else 0,
        "input_height": int(frames[0].shape[0]) if frames else 0,
        "repeat": repeat,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(latencies_ms),
        "p90_ms": percentile(latencies_ms, 90),
        "p95_ms": percentile(latencies_ms, 95),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "per_image_mean_ms": mean_ms / float(batch_size),
    }


def write_outputs(rows: list[dict[str, float | int]], output_csv: str, output_json: str) -> None:
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "batch_size",
        "input_width",
        "input_height",
        "engine_path",
        "repeat",
        "mean_ms",
        "median_ms",
        "p90_ms",
        "p95_ms",
        "min_ms",
        "max_ms",
        "per_image_mean_ms",
    ]
    with Path(output_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    Path(output_json).write_text(json.dumps(rows, indent=2), encoding="utf-8")


def plot_batch_curves(rows: list[dict[str, float | int]], output_csv: str) -> list[str]:
    csv_path = Path(output_csv)
    output_dir = csv_path.parent
    ordered = sorted(rows, key=lambda item: int(item["batch_size"]))
    x_values = [int(item["batch_size"]) for item in ordered]

    plots = [
        (
            [float(item["per_image_mean_ms"]) for item in ordered],
            "Per-image Latency (ms/image)",
            f"Batch Size K vs Per-image Latency ({ordered[0]['input_width']}x{ordered[0]['input_height']})",
            output_dir / f"{csv_path.stem}_per_image_latency.png",
        ),
        (
            [float(item["mean_ms"]) for item in ordered],
            "Batch Latency (ms)",
            f"Batch Size K vs Batch Latency ({ordered[0]['input_width']}x{ordered[0]['input_height']})",
            output_dir / f"{csv_path.stem}_batch_latency.png",
        ),
        (
            [1000.0 * int(item["batch_size"]) / float(item["mean_ms"]) for item in ordered],
            "Throughput (images/s)",
            f"Batch Size K vs Throughput ({ordered[0]['input_width']}x{ordered[0]['input_height']})",
            output_dir / f"{csv_path.stem}_throughput.png",
        ),
    ]

    saved_paths: list[str] = []
    for y_values, y_label, title, plot_path in plots:
        plt.figure(figsize=(7, 5))
        plt.plot(x_values, y_values, marker="o", linewidth=2)
        plt.xticks(x_values, [str(x) for x in x_values])
        plt.xlabel("Batch Size K")
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        saved_paths.append(str(plot_path))
    return saved_paths


def main() -> None:
    args = parse_args()
    max_batch = max(args.batch_sizes)
    frames = load_video_frames(
        args.video_dir,
        max(args.num_source_frames, max_batch),
        input_width=args.input_width,
        input_height=args.input_height,
    )

    cuda.init()
    try:
        cuda.Context.pop()
    except Exception:
        pass
    cfx = cuda.Device(0).make_context()

    try:
        rows = []
        for batch_size in args.batch_sizes:
            engine_path = None
            if args.engine_map is not None:
                engine_path = args.engine_map.get(batch_size)
                if engine_path is None:
                    raise ValueError(
                        f"Missing engine for K={batch_size}. engine_map keys={sorted(args.engine_map)}"
                    )
            else:
                engine_path = args.sr_model

            class SRArgs:
                sr_model_path = engine_path

            sr_model = None
            try:
                sr_model = SRBatchInfer(SRArgs(), cfx)
                result = benchmark_batch_size(
                    sr_model,
                    frames,
                    batch_size=batch_size,
                    warmup=args.warmup,
                    repeat=args.repeat,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Failed while benchmarking K={batch_size} with engine {engine_path}: {exc}"
                ) from exc
            finally:
                if sr_model is not None:
                    sr_model.close()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            result["engine_path"] = str(engine_path)
            rows.append(result)
            print(
                f"K={batch_size:<2d} "
                f"input={result['input_width']}x{result['input_height']} "
                f"mean={result['mean_ms']:.3f} ms "
                f"p50={result['median_ms']:.3f} ms "
                f"p90={result['p90_ms']:.3f} ms "
                f"per_img={result['per_image_mean_ms']:.3f} ms "
                f"engine={engine_path}"
            )
        write_outputs(rows, args.output_csv, args.output_json)
        plot_paths = plot_batch_curves(rows, args.output_csv)
        print(f"Saved CSV: {args.output_csv}")
        print(f"Saved JSON: {args.output_json}")
        for plot_path in plot_paths:
            print(f"Saved Plot: {plot_path}")
    finally:
        cfx.pop()


if __name__ == "__main__":
    main()
