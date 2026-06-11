from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pycuda.driver as cuda

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RACE.artifacts import HomographyArtifactLoader
from RACE.runtime import GOPOrchestrator, OnlineProposalGenerator, ProposalMatcher
from RACE.detector import Yolo11TRTDetector
from RACE.core import set_bin_size
from RACE.sr_model.sr_batch_infer import SRBatchInfer


BIN_AREA_640_360 = 640 * 360
BIN_AREA_256_256 = 256 * 256
SUPPORTED_SR_BATCH_SIZES = (1, 2, 3, 4, 5, 6, 7, 8)
THROUGHPUT_TAIL_FRAMES = 1500


class MultiBatchSRRunner:
    """Load one static SR TensorRT engine per exact batch size."""

    def __init__(self, engine_map: dict[int, str], cfx: cuda.Context):
        self.cfx = cfx
        self.engine_map = {int(batch_size): str(path) for batch_size, path in engine_map.items()}
        self.runners: dict[int, SRBatchInfer] = {}

        for batch_size in sorted(self.engine_map):
            engine_path = Path(self.engine_map[batch_size])
            if not engine_path.exists():
                raise FileNotFoundError(
                    f"Missing SR engine for batch size {batch_size}: {engine_path}"
                )

            class SRArgs:
                sr_model_path = str(engine_path)

            self.runners[batch_size] = SRBatchInfer(SRArgs(), cfx)

    def inference(self, imgs):
        """Dispatch the SR batch to the engine whose batch size matches K."""

        batch_size = int(imgs.shape[0])
        runner = self.runners.get(batch_size)
        if runner is None:
            raise RuntimeError(
                f"No SR engine loaded for batch size {batch_size}. "
                f"Available batch sizes: {sorted(self.runners)}"
            )
        return runner.inference(imgs)

    def close(self) -> None:
        for runner in self.runners.values():
            runner.close()
        self.runners.clear()


def parse_batch_sizes(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("batch sizes must be positive integers, e.g. 1,2,4,8")
    return values


def parse_class_ids(raw: str) -> tuple[int, ...] | None:
    value = raw.strip()
    if not value or value.lower() in {"all", "none", "*"}:
        return None
    class_ids = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not class_ids or any(class_id < 0 for class_id in class_ids):
        raise argparse.ArgumentTypeError("class ids must be non-negative integers, e.g. 2,5,7 or all")
    return tuple(sorted(set(class_ids)))


def infer_engine_map(sr_model_path: str, batch_sizes: tuple[int, ...]) -> dict[int, str]:
    """Infer sibling SR engine paths such as EDSR_256_b3.engine from one path."""

    path = Path(sr_model_path).resolve()
    match = re.match(r"^(?P<prefix>.+)_b(?P<batch>\d+)(?P<suffix>\.engine)$", path.name)
    if match:
        prefix = match.group("prefix")
        suffix = match.group("suffix")
        return {
            batch_size: str(path.parent / f"{prefix}_b{batch_size}{suffix}")
            for batch_size in batch_sizes
        }

    base_dir = Path(sr_model_path).resolve().parent
    return {
        batch_size: str(base_dir / f"EDSR_256_b{batch_size}.engine")
        for batch_size in batch_sizes
    }


def load_batch_latency_ms(latency_json_path: str) -> dict[int, float]:
    """Load measured SR latency by batch size for profile-based planning."""

    payload = json.loads(Path(latency_json_path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "batch_latency_ms" in payload:
        return {
            int(batch_size): float(latency)
            for batch_size, latency in payload["batch_latency_ms"].items()
        }

    rows = payload
    if isinstance(payload, dict) and "rows" in payload:
        rows = payload["rows"]
    if not isinstance(rows, list):
        raise ValueError(
            f"Unsupported SR latency JSON format in {latency_json_path}. "
            "Expected a benchmark row list or a dict with batch_latency_ms."
        )

    latency_by_batch: dict[int, float] = {}
    for row in rows:
        if "batch_size" not in row or "mean_ms" not in row:
            continue
        latency_by_batch[int(row["batch_size"])] = float(row["mean_ms"])
    if not latency_by_batch:
        raise ValueError(f"No batch latency rows found in {latency_json_path}")
    return latency_by_batch


def summarize_tail_throughput(summaries: list[dict], *, tail_frames: int = THROUGHPUT_TAIL_FRAMES) -> dict[str, float | int]:
    """Aggregate the last `tail_frames` into the summary throughput counters."""

    if not summaries:
        return {
            "gops": 0,
            "frames": 0,
            "camera_frames": 0,
            "cpu_seconds": 0.0,
            "gpu_seconds": 0.0,
            "overlap_seconds": 0.0,
            "sequential_seconds": 0.0,
            "transfer_stage_seconds": 0.0,
            "transfer_frame_copy_seconds": 0.0,
            "transfer_compare_seconds": 0.0,
            "video_fps_overlap": 0.0,
            "video_fps_sequential": 0.0,
        }

    max_frame_id = max(int(frame_id) for summary in summaries for frame_id in summary.get("frame_ids", []))
    cutoff_frame_id = max_frame_id - int(tail_frames) + 1
    selected = [
        summary for summary in summaries
        if summary.get("frame_ids") and max(int(frame_id) for frame_id in summary["frame_ids"]) >= cutoff_frame_id
    ]

    frame_count = sum(
        sum(1 for frame_id in summary.get("frame_ids", []) if int(frame_id) >= cutoff_frame_id)
        for summary in selected
    )
    camera_frame_count = sum(int(summary.get("detect_input_count", 0)) for summary in selected)
    cpu_seconds = 0.0
    gpu_seconds = 0.0
    transfer_stage_seconds = 0.0
    transfer_frame_copy_seconds = 0.0
    for summary in selected:
        frame_stage_seconds = float(summary.get("detect_frame_cpu_stage_seconds", 0.0))
        frame_gpu_copy_seconds = (
            float(summary.get("detect_frame_h2d_seconds", 0.0))
            + float(summary.get("detect_frame_gpu_convert_seconds", 0.0))
        )
        transfer_stage_seconds += frame_stage_seconds
        transfer_frame_copy_seconds += frame_gpu_copy_seconds
        cpu_seconds += (
            float(summary.get("cpu_total_seconds", 0.0))
            + frame_stage_seconds
        )
        gpu_seconds += (
            frame_gpu_copy_seconds
            + float(summary.get("sr_seconds", 0.0))
            + float(summary.get("detect_blend_seconds", 0.0))
            + float(summary.get("detect_seconds", 0.0))
        )
    # Current worker mode overlaps next-GOP proposal with this main-thread path,
    # but finalize/stage/GPU are still serialized for each GOP.
    overlap_seconds = cpu_seconds + gpu_seconds
    sequential_seconds = cpu_seconds + gpu_seconds
    transfer_compare_seconds = transfer_frame_copy_seconds

    return {
        "gops": len(selected),
        "frames": frame_count,
        "camera_frames": camera_frame_count,
        "cutoff_frame_id": cutoff_frame_id,
        "cpu_seconds": cpu_seconds,
        "gpu_seconds": gpu_seconds,
        "overlap_seconds": overlap_seconds,
        "sequential_seconds": sequential_seconds,
        "transfer_stage_seconds": transfer_stage_seconds,
        "transfer_frame_copy_seconds": transfer_frame_copy_seconds,
        "transfer_compare_seconds": transfer_compare_seconds,
        "video_fps_overlap": float(frame_count) / overlap_seconds if overlap_seconds > 0 else 0.0,
        "video_fps_sequential": float(frame_count) / sequential_seconds if sequential_seconds > 0 else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RACE GOP runtime pipeline")
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--homography_artifact", required=True)
    parser.add_argument(
        "--proposal_artifact",
        default=None,
        help="Optional output path to save online-generated proposal cache for debugging.",
    )
    parser.add_argument("--sr_model", required=True)
    parser.add_argument("--det_model", required=True)
    parser.add_argument(
        "--sr_latency_json",
        default="RACE/output_runtime_sr_batch_256/sr_latency.json",
        help="Benchmark JSON with mean SR latency per batch size. Used only with --sr_launch_policy profile.",
    )
    parser.add_argument(
        "--sr_launch_policy",
        choices=("exact", "profile"),
        default="exact",
        help="exact runs one engine matching the actual SR bin count; profile uses latency JSON to split batches.",
    )
    parser.add_argument("--output_dir", default="RACE/output_runtime")
    parser.add_argument(
        "--disable_save_outputs",
        action="store_true",
        help="Disable saving SR bin images and visualization frames for profiling.",
    )
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument(
        "--num_bins",
        type=int,
        default=4,
        help="Deprecated compatibility flag. Packing now opens bins on demand.",
    )
    parser.add_argument(
        "--max_sr_bins",
        type=int,
        default=8,
        help="Maximum number of SR bins admitted per GOP. Use 0 to disable the budget.",
    )
    parser.add_argument("--bin_width", type=int, default=256)
    parser.add_argument("--bin_height", type=int, default=256)
    parser.add_argument(
        "--sr_batch_sizes",
        type=parse_batch_sizes,
        default=SUPPORTED_SR_BATCH_SIZES,
        help="Comma-separated SR batch sizes to load, e.g. 1,2,4,8.",
    )
    parser.add_argument(
        "--disable_cross_camera_dedup",
        action="store_true",
        help="Run the object-level no-dedup baseline: each proposal is enhanced independently.",
    )
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument(
        "--class_ids",
        type=parse_class_ids,
        default=(2, 5, 7),
        help="Comma-separated detector COCO class ids to keep. Use `all` to disable class filtering.",
    )
    return parser.parse_args()
 

def main() -> None:
    """CLI entry point that wires artifacts, TRT engines, and the GOP runtime."""

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_bin_size(args.bin_width, args.bin_height)
    print(
        "[RACE Config] "
        f"bin_size={args.bin_width}x{args.bin_height} "
        f"sr_batch_sizes={list(args.sr_batch_sizes)} "
        f"sr_launch_policy={args.sr_launch_policy} "
        f"sr_model={args.sr_model} "
        f"det_conf={args.conf_thresh} "
        f"det_class_ids={args.class_ids if args.class_ids is not None else 'all'} "
        "frame_transfer=direct",
        flush=True,
    )

    homography_loader = HomographyArtifactLoader.from_file(args.homography_artifact)
    matcher = ProposalMatcher(homography_loader)
    camera_ids = homography_loader.cameras or sorted(
        {cam for pair in homography_loader.iter_pairs() for cam in (pair.src_cam, pair.ref_cam)}
    )
    proposal_generator = OnlineProposalGenerator(camera_ids)

    cuda.init()
    try:
        cuda.Context.pop()
    except Exception:
        pass
    cfx = cuda.Device(0).make_context()
    sr_engine_map = infer_engine_map(args.sr_model, args.sr_batch_sizes)
    sr_batch_latency_ms = None
    if args.sr_launch_policy == "profile":
        raw_sr_batch_latency_ms = load_batch_latency_ms(args.sr_latency_json)
        sr_batch_latency_ms = {
            batch_size: latency
            for batch_size, latency in raw_sr_batch_latency_ms.items()
            if batch_size in sr_engine_map
        }
        missing_latency = [batch_size for batch_size in args.sr_batch_sizes if batch_size not in sr_batch_latency_ms]
        if missing_latency:
            raise ValueError(
                f"Missing SR latency for batch sizes {missing_latency} in {args.sr_latency_json}"
            )

    sr_model = MultiBatchSRRunner(sr_engine_map, cfx)
    detector = Yolo11TRTDetector(args.det_model, cfx, conf_thresh=args.conf_thresh, class_ids=args.class_ids)

    orchestrator = GOPOrchestrator(
        video_dir=args.video_dir,
        matcher=matcher,
        sr_model=sr_model,
        detector=detector,
        output_dir=args.output_dir,
        proposal_generator=proposal_generator,
        proposal_cache_path=args.proposal_artifact,
        gop=args.gop,
        num_bins=args.num_bins,
        max_sr_bins=args.max_sr_bins if args.max_sr_bins > 0 else None,
        sr_batch_latency_ms=sr_batch_latency_ms,
        sr_launch_policy=args.sr_launch_policy,
        save_outputs=not args.disable_save_outputs,
        detection_device="cuda",
        sr_device="cuda",
        cross_camera_dedup=not args.disable_cross_camera_dedup,
    )

    try:
        summaries = orchestrator.run()

        total_raw_area = sum(int(summary.get("raw_proposal_area", 0)) for summary in summaries)
        total_dedup_area = sum(int(summary.get("dedup_candidate_area", 0)) for summary in summaries)
        total_placed_area = sum(int(summary.get("placed_candidate_area", 0)) for summary in summaries)
        total_raw_count = sum(int(summary.get("raw_proposal_count", 0)) for summary in summaries)
        total_dedup_count = sum(int(summary.get("dedup_candidate_count", 0)) for summary in summaries)
        total_placed_count = sum(int(summary.get("placed_candidate_count", 0)) for summary in summaries)
        total_raw_bins_640_360_per_gop = sum(int(summary.get("raw_bins_640x360", 0)) for summary in summaries)
        total_raw_bins_256_256_per_gop = sum(int(summary.get("raw_bins_256x256", 0)) for summary in summaries)
        total_dedup_bins_640_360_per_gop = sum(int(summary.get("dedup_bins_640x360", 0)) for summary in summaries)
        total_dedup_bins_256_256_per_gop = sum(int(summary.get("dedup_bins_256x256", 0)) for summary in summaries)
        total_reduced_bins_640_360_per_gop = total_raw_bins_640_360_per_gop - total_dedup_bins_640_360_per_gop
        total_reduced_bins_256_256_per_gop = total_raw_bins_256_256_per_gop - total_dedup_bins_256_256_per_gop
        opened_bin_counts = [int(summary.get("opened_bin_count", 0)) for summary in summaries]
        total_opened_bin_count = sum(opened_bin_counts)
        total_sr_canvas_pixels = sum(int(summary.get("sr_canvas_pixels", 0)) for summary in summaries)
        total_placed_pixels = sum(int(summary.get("placed_pixels", 0)) for summary in summaries)
        fill_ratio = (
            float(total_placed_pixels) / float(total_sr_canvas_pixels)
            if total_sr_canvas_pixels > 0 else 0.0
        )
        avg_opened_bin_count = (
            float(total_opened_bin_count) / float(len(opened_bin_counts))
            if opened_bin_counts else 0.0
        )
        cross_camera_dedup = all(bool(summary.get("cross_camera_dedup", True)) for summary in summaries)

        raw_bins_640_360 = int((total_raw_area + BIN_AREA_640_360 - 1) // BIN_AREA_640_360) if total_raw_area > 0 else 0
        raw_bins_256_256 = int((total_raw_area + BIN_AREA_256_256 - 1) // BIN_AREA_256_256) if total_raw_area > 0 else 0
        dedup_bins_640_360 = int((total_dedup_area + BIN_AREA_640_360 - 1) // BIN_AREA_640_360) if total_dedup_area > 0 else 0
        dedup_bins_256_256 = int((total_dedup_area + BIN_AREA_256_256 - 1) // BIN_AREA_256_256) if total_dedup_area > 0 else 0
        reduced_bins_640_360 = raw_bins_640_360 - dedup_bins_640_360
        reduced_bins_256_256 = raw_bins_256_256 - dedup_bins_256_256
        reduction_ratio = 0.0
        if total_raw_area > 0:
            reduction_ratio = 1.0 - (float(total_dedup_area) / float(total_raw_area))

        print(
            "[RACE Summary] "
            f"cross_camera_dedup={'on' if cross_camera_dedup else 'off'} "
            f"raw_proposals={total_raw_count} "
            f"raw_area={total_raw_area} "
            f"dedup_proposals={total_dedup_count} "
            f"dedup_area={total_dedup_area} "
            f"placed_proposals={total_placed_count} "
            f"placed_area={total_placed_area} "
            f"area_reduction={reduction_ratio:.2%}"
        )
        print(
            "[RACE Summary] "
            f"raw_bins_640x360={raw_bins_640_360} "
            f"raw_bins_256x256={raw_bins_256_256} "
            f"dedup_bins_640x360={dedup_bins_640_360} "
            f"dedup_bins_256x256={dedup_bins_256_256} "
            f"reduced_bins_640x360={reduced_bins_640_360} "
            f"reduced_bins_256x256={reduced_bins_256_256}"
        )
        print(
            "[RACE Summary Per-GOP Sum] "
            f"raw_bins_640x360={total_raw_bins_640_360_per_gop} "
            f"raw_bins_256x256={total_raw_bins_256_256_per_gop} "
            f"dedup_bins_640x360={total_dedup_bins_640_360_per_gop} "
            f"dedup_bins_256x256={total_dedup_bins_256_256_per_gop} "
            f"reduced_bins_640x360={total_reduced_bins_640_360_per_gop} "
            f"reduced_bins_256x256={total_reduced_bins_256_256_per_gop}"
        )
        print(
            "[RACE Summary Opened Bins] "
            f"total={total_opened_bin_count} "
            f"avg_per_gop={avg_opened_bin_count:.2f} "
            f"counts={opened_bin_counts}"
        )
        print(
            "[RACE Summary Packing Efficiency] "
            f"sr_canvas_pixels={total_sr_canvas_pixels} "
            f"placed_pixels={total_placed_pixels} "
            f"fill_ratio={fill_ratio:.2%}"
        )
        throughput = summarize_tail_throughput(summaries, tail_frames=THROUGHPUT_TAIL_FRAMES)
        print(
            "[RACE Summary Throughput Tail] "
            f"tail_frames={THROUGHPUT_TAIL_FRAMES} "
            f"cutoff_frame={int(throughput.get('cutoff_frame_id', 0))} "
            f"gops={int(throughput['gops'])} "
            f"frames={int(throughput['frames'])} "
            f"camera_frames={int(throughput['camera_frames'])} "
            f"cpu={float(throughput['cpu_seconds']) * 1000.0:.3f}ms "
            f"gpu={float(throughput['gpu_seconds']) * 1000.0:.3f}ms"
        )
        print(
            "[RACE Summary Throughput Tail] "
            f"overlap_time={float(throughput['overlap_seconds']) * 1000.0:.3f}ms "
            f"video_fps={float(throughput['video_fps_overlap']):.2f}"
        )
        print(
            "[RACE Summary Transfer Tail] "
            "mode=direct "
            f"stage={float(throughput['transfer_stage_seconds']) * 1000.0:.3f}ms "
            f"frame_copy={float(throughput['transfer_frame_copy_seconds']) * 1000.0:.3f}ms "
            f"compare_time={float(throughput['transfer_compare_seconds']) * 1000.0:.3f}ms"
        )
        print(
            "[RACE Summary Throughput Tail Sequential] "
            f"sequential_time={float(throughput['sequential_seconds']) * 1000.0:.3f}ms "
            f"video_fps={float(throughput['video_fps_sequential']):.2f}"
        )
    finally:
        try:
           cuda.Context.synchronize()
        except Exception:
            pass
        try:
            detector.close()
        except Exception as exc:
            print(f"[WARN] detector.close() failed: {exc}", flush=True)
        try:
            sr_model.close()
        except Exception as exc:
            print(f"[WARN] sr_model.close() failed: {exc}", flush=True)
        orchestrator.close()
        cfx.pop() 


if __name__ == "__main__":
    main()
