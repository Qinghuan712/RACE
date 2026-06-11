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


SRC_W, SRC_H = 640, 360


def parse_sizes(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("sizes must be positive integers, e.g. 64,128,192,256,320")
    return values


def parse_engine_map(raw: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError(
                "engine_map must look like 64:path/to/s64.engine,128:path/to/s128.engine"
            )
        key, value = item.split(":", 1)
        size = int(key.strip())
        engine_path = value.strip()
        if size <= 0 or not engine_path:
            raise argparse.ArgumentTypeError(
                "engine_map entries must use a positive size and non-empty path"
            )
        mapping[size] = engine_path
    if not mapping:
        raise argparse.ArgumentTypeError("engine_map cannot be empty")
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SR latency versus input size.")
    parser.add_argument("--video_dir", default="./dataset_preprocessing/aligned_videos_640")
    parser.add_argument("--sr_model", default=None, help="Single engine path used for all input sizes.")
    parser.add_argument(
        "--engine_map",
        type=parse_engine_map,
        default=None,
        help="Per-size engine map, e.g. 64:temp_model/EDSR_s64.engine,128:temp_model/EDSR_s128.engine",
    )
    parser.add_argument("--sizes", type=parse_sizes, default=parse_sizes("64,128,192,256,320"))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--num_source_frames", type=int, default=128)
    parser.add_argument("--output_csv", default="RACE/output_runtime_sr_input/sr_input_size_latency.csv")
    parser.add_argument("--output_json", default="RACE/output_runtime_sr_input/sr_input_size_latency.json")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
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


def load_video_frames(video_dir: str, limit: int) -> list[np.ndarray]:
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
                if frame.shape[1] != SRC_W or frame.shape[0] != SRC_H:
                    frame = cv2.resize(frame, (SRC_W, SRC_H), interpolation=cv2.INTER_LINEAR)
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


def make_batch(frames: list[np.ndarray], batch_size: int, input_size: int, offset: int) -> torch.Tensor:
    batch_frames = []
    for idx in range(batch_size):
        frame = frames[(offset + idx) % len(frames)]
        resized = cv2.resize(frame, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        batch_frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32))
    rgb = np.stack(batch_frames, axis=0)
    return torch.from_numpy(rgb).permute(0, 3, 1, 2).cuda().float().contiguous()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, pct))


def benchmark_input_size(
    sr_model: SRBatchInfer,
    frames: list[np.ndarray],
    batch_size: int,
    input_size: int,
    warmup: int,
    repeat: int,
) -> dict[str, float | int]:
    batch = make_batch(frames, batch_size=batch_size, input_size=input_size, offset=0)

    for _ in range(warmup):
        _ = sr_model.inference(batch)
    torch.cuda.synchronize()

    latencies_ms: list[float] = []
    for idx in range(repeat):
        batch = make_batch(
            frames,
            batch_size=batch_size,
            input_size=input_size,
            offset=idx * batch_size,
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        _ = sr_model.inference(batch)
        torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - started) * 1000.0)

    mean_ms = statistics.mean(latencies_ms)
    return {
        "input_size": input_size,
        "batch_size": batch_size,
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
        "input_size",
        "batch_size",
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


def plot_latency_curve(rows: list[dict[str, float | int]], output_csv: str) -> str:
    csv_path = Path(output_csv)
    plot_path = csv_path.with_name(csv_path.stem + "_plot.png")

    ordered = sorted(rows, key=lambda item: int(item["input_size"]))
    x_values = [int(item["input_size"]) * int(item["input_size"]) for item in ordered]
    x_labels = [f"{int(item['input_size'])}x{int(item['input_size'])}" for item in ordered]
    y_values = [float(item["mean_ms"]) for item in ordered]
    batch_size = int(ordered[0]["batch_size"]) if ordered else 1

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker="o", linewidth=2)
    plt.xticks(x_values, x_labels, rotation=30)
    plt.xlabel("Input (H*W)")
    plt.ylabel("Latency (ms)")
    plt.title(f"SR Latency vs Input Size (K={batch_size})")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return str(plot_path)


def main() -> None:
    args = parse_args()
    frames = load_video_frames(args.video_dir, max(args.num_source_frames, args.batch_size))

    cuda.init()
    try:
        cuda.Context.pop()
    except Exception:
        pass
    cfx = cuda.Device(0).make_context()

    try:
        rows = []
        for input_size in args.sizes:
            if args.engine_map is not None:
                engine_path = args.engine_map.get(input_size)
                if engine_path is None:
                    raise ValueError(
                        f"Missing engine for input_size={input_size}. engine_map keys={sorted(args.engine_map)}"
                    )
            else:
                engine_path = args.sr_model

            class SRArgs:
                sr_model_path = engine_path

            sr_model = None
            try:
                sr_model = SRBatchInfer(SRArgs(), cfx)
                result = benchmark_input_size(
                    sr_model,
                    frames,
                    batch_size=args.batch_size,
                    input_size=input_size,
                    warmup=args.warmup,
                    repeat=args.repeat,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Failed while benchmarking input_size={input_size} with engine {engine_path}: {exc}"
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
                f"size={input_size:<3d} "
                f"K={args.batch_size:<2d} "
                f"mean={result['mean_ms']:.3f} ms "
                f"p50={result['median_ms']:.3f} ms "
                f"p90={result['p90_ms']:.3f} ms "
                f"per_img={result['per_image_mean_ms']:.3f} ms "
                f"engine={engine_path}"
            )

        write_outputs(rows, args.output_csv, args.output_json)
        plot_path = plot_latency_curve(rows, args.output_csv)
        print(f"Saved CSV: {args.output_csv}")
        print(f"Saved JSON: {args.output_json}")
        print(f"Saved Plot: {plot_path}")
    finally:
        cfx.pop()


if __name__ == "__main__":
    main()
