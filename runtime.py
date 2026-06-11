from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import RACE.core as race_core
from RACE.artifacts import HomographyArtifactLoader, Proposal, save_proposal_artifact
from RACE.core import CandidatePatch, PlacementEntry, blend_back_frames_torch, compute_importance, compute_view_quality, normalize_candidate_importance, optimize_launch_plan, pack_objects, run_sr_batch, select_best_view, select_candidates_under_bin_budget, tensor_to_bgr_image
from RACE.proposal_generation import DEFAULT_CONFIG as PROPOSAL_DEFAULT_CONFIG
from RACE.proposal_generation import _consolidate_boxes, _split_merged_blob


@contextmanager
def nvtx_range(name: str):
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


def _fmt_ms(seconds: float | int | None) -> str:
    return f"{float(seconds or 0.0) * 1000.0:.1f}ms"


def _copy_detect_targets_to_gpu(
    detect_targets: list[tuple[int, str, np.ndarray, dict[str, int] | None]],
    *,
    device: str | torch.device,
    proposal_pool: "ProposalWorkerPool | None" = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Copy LR detect frames from shared memory views to one CUDA NCHW batch.

    The direct mode keeps decoded frames in per-camera shared-memory slots. If a
    run of targets is contiguous in the same camera slot, we copy that slice as
    one NumPy view instead of stacking frame-by-frame on the CPU.
    """

    device = torch.device(device)
    timing = {
        "cpu_stage_seconds": 0.0,
        "h2d_seconds": 0.0,
        "gpu_convert_seconds": 0.0,
    }
    if not detect_targets:
        return torch.empty((0, 3, 0, 0), device=device, dtype=torch.float32), timing

    first = detect_targets[0][2]
    if first.ndim != 3 or first.shape[-1] != 3:
        raise ValueError(f"Expected HWC BGR frames, got shape={first.shape}")
    count = len(detect_targets)

    cpu_started = time.time()
    host_batches: list[np.ndarray] = []
    run_start = 0
    while run_start < count:
        run_cam = detect_targets[run_start][1]
        run_end = run_start + 1
        while run_end < count and detect_targets[run_end][1] == run_cam:
            run_end += 1
        host_batch: np.ndarray | None = None
        if proposal_pool is not None:
            ref0 = detect_targets[run_start][3]
            if ref0 is not None:
                slot = int(ref0["slot"])
                offset0 = int(ref0["offset"])
                contiguous = True
                for idx in range(run_start, run_end):
                    _, cam_id, frame_lr, frame_ref = detect_targets[idx]
                    if frame_lr.shape != first.shape:
                        raise ValueError(f"Expected same frame shape in GOP, got {frame_lr.shape} and {first.shape}")
                    if (
                        frame_ref is None
                        or cam_id != run_cam
                        or int(frame_ref["slot"]) != slot
                        or int(frame_ref["offset"]) != offset0 + (idx - run_start)
                    ):
                        contiguous = False
                        break
                if contiguous:
                    host_batch = proposal_pool.frame_block(
                        run_cam,
                        slot,
                        offset0,
                        offset0 + (run_end - run_start),
                    )
        if host_batch is None:
            host_batch = np.stack(
                [np.ascontiguousarray(frame_lr) for _, _, frame_lr, _ in detect_targets[run_start:run_end]],
                axis=0,
            )
        host_batches.append(np.ascontiguousarray(host_batch))
        run_start = run_end
    timing["cpu_stage_seconds"] = time.time() - cpu_started
    h2d_started = time.time()
    gpu_batches = [torch.from_numpy(host_batch).to(device=device) for host_batch in host_batches]
    batch_u8 = gpu_batches[0] if len(gpu_batches) == 1 else torch.cat(gpu_batches, dim=0)
    timing["h2d_seconds"] = time.time() - h2d_started

    convert_started = time.time()
    frame_batch = batch_u8.permute(0, 3, 1, 2).float().contiguous()
    timing["gpu_convert_seconds"] = time.time() - convert_started
    return frame_batch, timing


def _materialize_bins_on_gpu(
    detect_targets: list[tuple[int, str, np.ndarray, dict[str, int] | None]],
    detect_frame_batch: torch.Tensor,
    placements: list[PlacementEntry],
    *,
    device: str | torch.device,
    return_timing: bool = False,
) -> list[torch.Tensor] | tuple[list[torch.Tensor], dict[str, float]]:
    """Build SR input bins by cropping candidate boxes from full GPU frames."""

    timing = {
        "bin_init_seconds": 0.0,    
        "patch_resize_seconds": 0.0,
        "patch_paste_seconds": 0.0,
    }
    if not placements:
        empty: list[torch.Tensor] = []
        return (empty, timing) if return_timing else empty

    started = time.time()
    device = torch.device(device)
    bins = [
        torch.zeros((3, int(race_core.BIN_H), int(race_core.BIN_W)), device=device, dtype=torch.float32)
        for _ in range(max(entry.bin_idx for entry in placements) + 1)
    ]
    target_to_idx = {
        (frame_id, cam_id): idx
        for idx, (frame_id, cam_id, _, _) in enumerate(detect_targets)
    }
    timing["bin_init_seconds"] = time.time() - started

    for entry in placements:
        source_idx = target_to_idx.get((entry.frame_id, entry.cam_id))
        if source_idx is None or entry.bin_idx >= len(bins):
            continue
        frame_tensor = detect_frame_batch[source_idx]
        x, y, w, h = entry.orig_bbox
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(int(frame_tensor.shape[2]), int(x + w))
        y2 = min(int(frame_tensor.shape[1]), int(y + h))
        if x2 <= x1 or y2 <= y1:
            continue
        patch = frame_tensor[:, y1:y2, x1:x2]
        if patch.numel() == 0:
            continue
        if patch.shape[1] != entry.bh or patch.shape[2] != entry.bw:
            resize_started = time.time()
            patch = F.interpolate(
                patch.unsqueeze(0),
                size=(int(entry.bh), int(entry.bw)),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            timing["patch_resize_seconds"] += time.time() - resize_started
        paste_started = time.time()
        bins[entry.bin_idx][:, entry.by:entry.by + entry.bh, entry.bx:entry.bx + entry.bw] = patch
        timing["patch_paste_seconds"] += time.time() - paste_started

    if return_timing:
        return bins, timing
    return bins


RUNTIME_DEFAULT_CONFIG = {
    "cluster_min_f1": 0.5,
    "cluster_score_floor": 0.0,
    "proposal_warmup_frames": 300,
    "proposal_history": 500,
    "proposal_var_threshold": 60.0,
    "proposal_min_area": 200,
    "proposal_detect_shadows": False,
    "proposal_use_grayscale": True,
}

RAW_BIN_AREA_640_360 = 640 * 360
RAW_BIN_AREA_256_256 = 256 * 256


@dataclass
class Cluster:
    cluster_id: str
    frame_id: int
    proposals: list[Proposal]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedGOP:
    gop_index: int
    frame_ids: list[int]
    active_frame_ids: list[int]
    frames: dict[int, dict[str, np.ndarray]]
    frame_refs: dict[int, dict[str, dict[str, int]]]
    proposals: dict[int, dict[str, list[Proposal]]]
    clusters: dict[int, list[Cluster]]
    candidates: list[CandidatePatch]
    bins: list[Any]
    placements: list[PlacementEntry]
    deferred: list[CandidatePatch]
    cpu_stats: dict[str, float]
    area_stats: dict[str, float | int]
    worker_stats: dict[str, dict[str, float]]


@dataclass
class WorkerGOPJob:
    job_id: str
    gop_index: int
    start_frame: int
    max_frame: int | None
    submitted_at: float


@dataclass
class RawGOPInput:
    gop_index: int
    frame_ids: list[int]
    active_frame_ids: list[int]
    frames: dict[int, dict[str, np.ndarray]]
    frame_refs: dict[int, dict[str, dict[str, int]]]
    proposals: dict[int, dict[str, list[Proposal]]]
    raw_proposal_count: int
    raw_proposal_area: int
    decode_seconds: float
    proposal_seconds: float
    proposal_bgsub_seconds: float
    proposal_morphology_seconds: float
    proposal_contour_seconds: float
    proposal_consolidate_seconds: float
    proposal_split_seconds: float
    proposal_materialize_seconds: float
    worker_stats: dict[str, dict[str, float]]
    collect_seconds: float = 0.0
    proposal_sync_overhead_seconds: float = 0.0


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    _, _, w, h = bbox
    return max(0, int(w)) * max(0, int(h))


def _estimate_bin_count(total_area: float, bin_area: int) -> int:
    if total_area <= 0:
        return 0
    return int(np.ceil(float(total_area) / float(bin_area)))


def _candidate_snapshot(candidate: CandidatePatch) -> dict[str, Any]:
    x, y, w, h = candidate.orig_bbox
    return {
        "cluster_id": candidate.cluster_id,
        "camera_id": candidate.cam_id,
        "frame_id": candidate.frame_id,
        "bbox": [int(x), int(y), int(w), int(h)],
        "patch_width": int(candidate.w),
        "patch_height": int(candidate.h),
        "importance": float(candidate.importance),
        "metadata": candidate.metadata,
    }


class OnlineProposalGenerator:
    """Online background-subtraction proposal generator shared by all modes."""

    def __init__(
        self,
        camera_ids: list[str],
        *,
        cfg: dict[str, Any] | None = None,
    ):
        base_cfg = dict(PROPOSAL_DEFAULT_CONFIG)
        base_cfg.update(
            {
                "gt_dir": "",
                "split_debug": False,
                "split_debug_paper_mode": False,
                "warmup_frames": RUNTIME_DEFAULT_CONFIG["proposal_warmup_frames"],
                "history": RUNTIME_DEFAULT_CONFIG["proposal_history"],
                "var_threshold": RUNTIME_DEFAULT_CONFIG["proposal_var_threshold"],
                "min_area": RUNTIME_DEFAULT_CONFIG["proposal_min_area"],
                "detect_shadows": RUNTIME_DEFAULT_CONFIG["proposal_detect_shadows"],
                "use_grayscale": RUNTIME_DEFAULT_CONFIG["proposal_use_grayscale"],
            }
        )
        if cfg:
            base_cfg.update(cfg)
        self.cfg = base_cfg
        self.camera_ids = list(camera_ids)
        self.back_subs = {
            cam_id: cv2.createBackgroundSubtractorMOG2(
                history=self.cfg["history"],
                varThreshold=self.cfg["var_threshold"],
                detectShadows=self.cfg["detect_shadows"],
            )
            for cam_id in self.camera_ids
        }
        self.close_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.cfg["kernel_close_size"], self.cfg["kernel_close_size"]),
            )
            if self.cfg["kernel_close_size"] > 0 else None
        )
        self.open_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.cfg["kernel_open_size"], self.cfg["kernel_open_size"]),
            )
            if self.cfg["kernel_open_size"] > 0 else None
        )
        self.processed_frames = 0
        self.generated: list[Proposal] = []

    def process_frame(
        self,
        frame_id: int,
        frames: dict[str, np.ndarray],
    ) -> tuple[dict[str, list[Proposal]], dict[str, Any]]:
        """Update MOG2 state and emit per-camera proposal boxes for one frame."""

        self.processed_frames += 1
        if self.processed_frames <= int(self.cfg["warmup_frames"]):
            learning_rate = float(self.cfg["warmup_learning_rate"])
            record = False
        else:
            learning_rate = float(self.cfg["detection_learning_rate"])
            record = True

        proposals_by_camera: dict[str, list[Proposal]] = {}
        num_boxes = 0
        bgsub_time = 0.0
        gray_time = 0.0
        mog2_time = 0.0
        morphology_time = 0.0
        contour_time = 0.0
        consolidate_time = 0.0
        split_time = 0.0
        materialize_time = 0.0
        for cam_id in self.camera_ids:
            frame = frames.get(cam_id)
            if frame is None:
                continue
            t0 = time.time()
            bgsub_input = frame
            if self.cfg.get("use_grayscale", False):
                t_gray = time.time()
                bgsub_input = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_time += time.time() - t_gray
            t_mog2 = time.time()
            fg_mask = self.back_subs[cam_id].apply(bgsub_input, learningRate=learning_rate)
            mog2_time += time.time() - t_mog2
            bgsub_time += time.time() - t0

            t0 = time.time()
            if self.cfg["detect_shadows"]:
                _, fg_mask = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
            if self.close_kernel is not None:
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.close_kernel)
            if self.open_kernel is not None:
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.open_kernel)
            morphology_time += time.time() - t0

            if not record:
                proposals_by_camera[cam_id] = []
                continue

            t0 = time.time()
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_time += time.time() - t0

            boxes = []
            for cnt in contours:
                if cv2.contourArea(cnt) < self.cfg["min_area"]:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                if h <= 0:
                    continue
                aspect_ratio = w / float(h)
                if not (self.cfg["min_aspect_ratio"] < aspect_ratio < self.cfg["max_aspect_ratio"]):
                    continue
                boxes.append((x, y, w, h))

            t0 = time.time()
            boxes = _consolidate_boxes(boxes, self.cfg)
            consolidate_time += time.time() - t0

            tagged = []
            t0 = time.time()
            for box in boxes:
                tagged.extend(
                    _split_merged_blob(
                        box,
                        fg_mask,
                        self.cfg,
                        prior=None,
                        stats=None,
                        allow=("h", "v"),
                        debug_list=None,
                    )
                )
            split_time += time.time() - t0

            proposals = []
            t0 = time.time()
            for idx, (bbox, tag) in enumerate(tagged):
                x, y, w, h = bbox
                proposal = Proposal(
                    proposal_id=f"{cam_id}_f{frame_id:06d}_p{idx:04d}",
                    camera_id=cam_id,
                    frame_id=frame_id,
                    bbox=(int(x), int(y), int(w), int(h)),
                    score=float(w * h),
                    source_frame=frame_id,
                    metadata={"split_tag": tag},
                )
                proposals.append(proposal)
                self.generated.append(proposal)
            materialize_time += time.time() - t0
            proposals_by_camera[cam_id] = proposals
            num_boxes += len(proposals)

        return proposals_by_camera, {
            "record": record,
            "learning_rate": learning_rate,
            "processed_frames": self.processed_frames,
            "num_proposals": num_boxes,
            "bgsub_seconds": bgsub_time,
            "gray_seconds": gray_time,
            "mog2_seconds": mog2_time,
            "morphology_seconds": morphology_time,
            "contour_seconds": contour_time,
            "consolidate_seconds": consolidate_time,
            "split_seconds": split_time,
            "materialize_seconds": materialize_time,
        }


def _camera_proposal_worker_main(
    camera_id: str,
    video_path: str,
    cfg: dict[str, Any],
    task_queue: Any,
    result_queue: Any,
    shm_name: str,
    shm_shape: tuple[int, int, int, int, int],
    shm_dtype: str,
) -> None:
    """Worker process loop for one camera.

    Each worker decodes its GOP, writes raw BGR frames into shared memory, then
    runs proposal generation on the decoded frame. The main process receives
    only proposal metadata plus shared-memory slot/offset references.
    """

    
    cv2.setNumThreads(1)
    cv2.setUseOptimized(True)
    cv2.ocl.setUseOpenCL(False)
    
    generator_cfg = dict(cfg)
    generator_cfg["proposal_camera_workers"] = 1
    generator = OnlineProposalGenerator([camera_id], cfg=generator_cfg)
    frame_shm = shared_memory.SharedMemory(name=shm_name)
    frame_buffer = np.ndarray(shm_shape, dtype=np.dtype(shm_dtype), buffer=frame_shm.buf)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        result_queue.put(
            {
                "kind": "error",
                "camera_id": camera_id,
                "message": f"Failed to open video for {camera_id}: {video_path}",
            }
        )
        return

    try:
        while True:
            task = task_queue.get()
            kind = task.get("kind")
            if kind == "close":
                break
            if kind != "prepare":
                continue

            job_id = str(task["job_id"])
            start_frame = int(task["start_frame"])
            gop = int(task["gop"])
            slot = int(task["slot"])
            max_frame = task.get("max_frame")
            if max_frame is not None:
                max_frame = int(max_frame)
            worker_started = time.time()

            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame - 1))
            frame_ids: list[int] = []
            frame_refs_by_frame: dict[int, dict[str, Any]] = {}
            proposals_by_frame: dict[int, list[Proposal]] = {}
            recorded_frame_ids: list[int] = []
            decode_time = 0.0
            shm_write_time = 0.0
            proposal_process_time = 0.0
            bgsub_time = 0.0
            gray_time = 0.0
            mog2_time = 0.0
            morphology_time = 0.0
            contour_time = 0.0
            consolidate_time = 0.0
            split_time = 0.0
            materialize_time = 0.0

            for offset in range(gop):
                frame_id = start_frame + offset
                if max_frame is not None and frame_id > max_frame:
                    break

                t0 = time.time()
                ret, frame = cap.read()
                decode_time += time.time() - t0
                if not ret or frame is None:
                    break

                frame_ids.append(frame_id)
                if frame.shape != tuple(shm_shape[2:]):
                    raise RuntimeError(
                        f"Unexpected frame shape for {camera_id}: got {frame.shape}, expected {tuple(shm_shape[2:])}"
                    )
                t0 = time.time()
                frame_buffer[slot, offset] = frame
                shm_write_time += time.time() - t0
                frame_refs_by_frame[frame_id] = {"slot": slot, "offset": offset}

                t0 = time.time()
                # Proposal generation consumes the same decoded frame that was
                # just published to shared memory; there is no extra decode.
                frame_props, frame_stats = generator.process_frame(frame_id, {camera_id: frame})
                proposal_process_time += time.time() - t0
                if bool(frame_stats.get("record", False)):
                    recorded_frame_ids.append(frame_id)
                proposals_by_frame[frame_id] = frame_props.get(camera_id, [])
                bgsub_time += float(frame_stats.get("bgsub_seconds", 0.0))
                gray_time += float(frame_stats.get("gray_seconds", 0.0))
                mog2_time += float(frame_stats.get("mog2_seconds", 0.0))
                morphology_time += float(frame_stats.get("morphology_seconds", 0.0))
                contour_time += float(frame_stats.get("contour_seconds", 0.0))
                consolidate_time += float(frame_stats.get("consolidate_seconds", 0.0))
                split_time += float(frame_stats.get("split_seconds", 0.0))
                materialize_time += float(frame_stats.get("materialize_seconds", 0.0))

            frame_bytes = len(frame_ids) * int(np.prod(shm_shape[2:])) * np.dtype(shm_dtype).itemsize
            payload = {
                "kind": "prepared",
                "job_id": job_id,
                "camera_id": camera_id,
                "frame_ids": frame_ids,
                "recorded_frame_ids": recorded_frame_ids,
                "frame_refs_by_frame": frame_refs_by_frame,
                "proposals_by_frame": proposals_by_frame,
                "stats": {
                    "decode_seconds": decode_time,
                    "shm_write_seconds": shm_write_time,
                    "proposal_process_seconds": proposal_process_time,
                    "worker_total_seconds": time.time() - worker_started,
                    "frame_bytes": frame_bytes,
                    "bgsub_seconds": bgsub_time,
                    "gray_seconds": gray_time,
                    "mog2_seconds": mog2_time,
                    "morphology_seconds": morphology_time,
                    "contour_seconds": contour_time,
                    "consolidate_seconds": consolidate_time,
                    "split_seconds": split_time,
                    "materialize_seconds": materialize_time,
                },
            }
            put_started = time.time()
            result_queue.put(payload)
            queue_put_time = time.time() - put_started
            result_queue.put(
                {
                    "kind": "put_stats",
                    "job_id": job_id,
                    "camera_id": camera_id,
                    "queue_put_seconds": queue_put_time,
                    "worker_total_with_put_seconds": time.time() - worker_started,
                }
            )
    finally:
        frame_shm.close()
        cap.release()


class ProposalWorkerPool:
    """Owns one proposal process and two shared-memory GOP slots per camera."""

    NUM_FRAME_SLOTS = 2

    def __init__(
        self,
        camera_ids: list[str],
        *,
        video_dir: str,
        cfg: dict[str, Any],
        gop: int,
    ):
        self.camera_ids = list(camera_ids)
        self.video_dir = str(video_dir)
        self.gop = int(gop)
        self._ctx = mp.get_context("spawn")
        self._task_queues: dict[str, Any] = {}
        self._result_queues: dict[str, Any] = {}
        self._processes: dict[str, mp.Process] = {}
        self._submit_count = 0
        self._frame_shms: dict[str, shared_memory.SharedMemory] = {}
        self._frame_shapes: dict[str, tuple[int, int, int, int, int]] = {}
        self._frame_arrays: dict[str, np.ndarray] = {}
        frame_dtype = np.dtype(np.uint8)

        for cam_id in self.camera_ids:
            video_path = os.path.join(self.video_dir, f"{cam_id}_aligned.avi")
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Missing video for {cam_id}: {video_path}")
            probe = cv2.VideoCapture(video_path)
            if not probe.isOpened():
                raise RuntimeError(f"Failed to open video for {cam_id}: {video_path}")
            frame_w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            frame_h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
            probe.release()
            shm_shape = (self.NUM_FRAME_SLOTS, self.gop, frame_h, frame_w, 3)
            frame_shm = shared_memory.SharedMemory(create=True, size=int(np.prod(shm_shape)) * frame_dtype.itemsize)
            self._frame_shms[cam_id] = frame_shm
            self._frame_shapes[cam_id] = shm_shape
            self._frame_arrays[cam_id] = np.ndarray(shm_shape, dtype=frame_dtype, buffer=frame_shm.buf)
            task_queue = self._ctx.Queue(maxsize=2)
            result_queue = self._ctx.Queue(maxsize=2)
            process = self._ctx.Process(
                target=_camera_proposal_worker_main,
                args=(
                    cam_id,
                    video_path,
                    dict(cfg),
                    task_queue,
                    result_queue,
                    frame_shm.name,
                    shm_shape,
                    frame_dtype.name,
                ),
                daemon=True,
            )
            process.start()
            self._task_queues[cam_id] = task_queue
            self._result_queues[cam_id] = result_queue
            self._processes[cam_id] = process

    def submit(self, job_id: str, *, start_frame: int, max_frame: int | None = None) -> None:
        """Assign a GOP to every camera worker, rotating between shm slots."""

        slot = self._submit_count % self.NUM_FRAME_SLOTS
        self._submit_count += 1
        task = {
            "kind": "prepare",
            "job_id": str(job_id),
            "start_frame": int(start_frame),
            "max_frame": int(max_frame) if max_frame is not None else None,
            "gop": self.gop,
            "slot": slot,
        }
        for cam_id in self.camera_ids:
            self._task_queues[cam_id].put(task)

    def frame_view(self, camera_id: str, slot: int, offset: int) -> np.ndarray:
        """Return a single decoded BGR frame view from shared memory."""

        return self._frame_arrays[camera_id][int(slot), int(offset)]

    def frame_block(self, camera_id: str, slot: int, offset_start: int, offset_end: int) -> np.ndarray:
        """Return a contiguous GOP slice for one camera from shared memory."""

        return self._frame_arrays[camera_id][int(slot), int(offset_start):int(offset_end)]

    def collect(self, job_id: str) -> dict[str, dict[str, Any]]:
        """Collect all camera payloads for one GOP and attach queue timings."""

        results: dict[str, dict[str, Any]] = {}
        get_total_seconds = 0.0
        for cam_id in self.camera_ids:
            get_started = time.time()
            payload = self._result_queues[cam_id].get()
            get_seconds = time.time() - get_started
            get_total_seconds += get_seconds
            payload.setdefault("stats", {})["queue_get_seconds"] = get_seconds
            if payload.get("kind") == "error":
                raise RuntimeError(payload.get("message", f"Worker failure for camera {cam_id}"))
            if payload.get("kind") != "prepared":
                raise RuntimeError(f"Unexpected worker payload for camera {cam_id}: {payload}")
            if str(payload.get("job_id")) != str(job_id):
                raise RuntimeError(
                    f"Mismatched worker job for camera {cam_id}: expected {job_id}, got {payload.get('job_id')}"
                )
            stats_started = time.time()
            put_stats = self._result_queues[cam_id].get()
            stats_get_seconds = time.time() - stats_started
            get_total_seconds += stats_get_seconds
            if put_stats.get("kind") != "put_stats":
                raise RuntimeError(f"Unexpected worker put-stats payload for camera {cam_id}: {put_stats}")
            if str(put_stats.get("job_id")) != str(job_id):
                raise RuntimeError(
                    f"Mismatched worker put-stats job for camera {cam_id}: expected {job_id}, got {put_stats.get('job_id')}"
                )
            payload["stats"]["queue_put_seconds"] = float(put_stats.get("queue_put_seconds", 0.0))
            payload["stats"]["queue_put_stats_get_seconds"] = stats_get_seconds
            payload["stats"]["worker_total_with_put_seconds"] = float(put_stats.get("worker_total_with_put_seconds", 0.0))
            results[cam_id] = payload
        for payload in results.values():
            payload.setdefault("stats", {})["collect_total_seconds"] = get_total_seconds
        return results

    def close(self) -> None:
        for cam_id in self.camera_ids:
            queue = self._task_queues.get(cam_id)
            if queue is None:
                continue
            try:
                queue.put({"kind": "close"})
            except Exception:
                pass
        for process in self._processes.values():
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        self._processes.clear()
        self._task_queues.clear()
        self._result_queues.clear()
        for frame_shm in self._frame_shms.values():
            try:
                frame_shm.close()
            except Exception:
                pass
            try:
                frame_shm.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass
        self._frame_shms.clear()
        self._frame_shapes.clear()
        self._frame_arrays.clear()


class _UnionFind:
    def __init__(self, nodes: list[tuple[str, str]]):
        self.parent = {node: node for node in nodes}
        self.cameras = {node: {node[0]} for node in nodes}

    def find(self, node: tuple[str, str]) -> tuple[str, str]:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def try_union(self, a: tuple[str, str], b: tuple[str, str]) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.cameras[ra] & self.cameras[rb]:
            return False
        if len(self.cameras[ra]) < len(self.cameras[rb]):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.cameras[ra] = self.cameras[ra] | self.cameras[rb]
        return True

    def components(self) -> list[list[tuple[str, str]]]:
        grouped = defaultdict(list)
        for node in self.parent:
            grouped[self.find(node)].append(node)
        return list(grouped.values())


def _point_in_hull(point: tuple[float, float], hull: np.ndarray | None, margin: float) -> bool:
    if hull is None:
        return True
    dist = cv2.pointPolygonTest(hull, (float(point[0]), float(point[1])), measureDist=True)
    return dist >= -margin


def _project_point(H: np.ndarray, point: tuple[float, float]) -> np.ndarray:
    vec = np.array([point[0], point[1], 1.0], dtype=np.float32)
    proj = H @ vec
    return proj[:2] / proj[2]


class ProposalMatcher:
    """Cluster same-object proposals across cameras using homography pairs."""

    def __init__(
        self,
        homography_loader: HomographyArtifactLoader,
        *,
        min_pair_f1: float = RUNTIME_DEFAULT_CONFIG["cluster_min_f1"],
        score_floor: float = RUNTIME_DEFAULT_CONFIG["cluster_score_floor"],
    ):
        self.homography_loader = homography_loader
        self.min_pair_f1 = min_pair_f1
        self.score_floor = score_floor
        self.strong_pairs = [
            pair for pair in homography_loader.iter_pairs()
            if pair.pair_f1 >= min_pair_f1 and pair.tau > 0
        ]

    def match_frame(
        self,
        frame_id: int,
        proposals_by_camera: dict[str, list[Proposal]],
    ) -> list[Cluster]:
        """Build cross-camera proposal clusters for one frame.

        Candidate edges are scored by projected bottom-center distance. Union
        find then greedily merges high-score edges while keeping at most one
        proposal from each camera in a cluster.
        """

        nodes = []
        proposal_by_key: dict[tuple[str, str], Proposal] = {}
        for cam_id, proposals in proposals_by_camera.items():
            for proposal in proposals:
                key = (cam_id, proposal.proposal_id)
                nodes.append(key)
                proposal_by_key[key] = proposal
        if not nodes:
            return []

        edges: list[tuple[float, float, tuple[str, str], tuple[str, str]]] = []
        for pair in self.strong_pairs:
            src_props = proposals_by_camera.get(pair.src_cam, [])
            ref_props = proposals_by_camera.get(pair.ref_cam, [])
            if not src_props or not ref_props:
                continue

            src_filtered = [
                proposal for proposal in src_props
                if _point_in_hull(proposal.bottom_center, pair.hull_src, pair.margin)
            ]
            ref_filtered = [
                proposal for proposal in ref_props
                if _point_in_hull(proposal.bottom_center, pair.hull_ref, pair.margin)
            ]
            if not src_filtered or not ref_filtered:
                continue

            projected = {
                proposal.proposal_id: _project_point(pair.H, proposal.bottom_center)
                for proposal in src_filtered
            }
            for src_proposal in src_filtered:
                proj = projected[src_proposal.proposal_id]
                for ref_proposal in ref_filtered:
                    ref_pt = np.array(ref_proposal.bottom_center, dtype=np.float32)
                    distance = float(np.linalg.norm(proj - ref_pt))
                    if distance > pair.tau:
                        continue
                    score = 1.0 - distance / pair.tau if pair.tau > 0 else 0.0
                    if score < self.score_floor:
                        continue
                    edges.append(
                        (
                            score,
                            distance,
                            (src_proposal.camera_id, src_proposal.proposal_id),
                            (ref_proposal.camera_id, ref_proposal.proposal_id),
                        )
                    )

        edges.sort(key=lambda item: (-item[0], item[1], item[2][1], item[3][1]))
        uf = _UnionFind(nodes)
        for score, distance, u, v in edges:
            _ = score, distance
            uf.try_union(u, v)

        clusters = []
        for cluster_idx, component in enumerate(uf.components()):
            proposals = [proposal_by_key[node] for node in component]
            clusters.append(
                Cluster(
                    cluster_id=f"f{frame_id:06d}_c{cluster_idx:04d}",
                    frame_id=frame_id,
                    proposals=sorted(proposals, key=lambda p: (p.camera_id, p.proposal_id)),
                )
            )
        return clusters


class DetectionRunner:
    """Thin batching wrapper around the TensorRT detector."""

    def __init__(self, detector: Any, *, device: str | torch.device | None = None):
        self.detector = detector
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.max_batch = max(1, int(getattr(detector, "max_batch", 1) or 1))
        self.opt_batch = max(1, int(getattr(detector, "opt_batch", self.max_batch) or self.max_batch))

    def _to_tensor(self, frame_hr: np.ndarray | torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(frame_hr):
            tensor = frame_hr
            if tensor.ndim == 3 and tensor.shape[0] != 3 and tensor.shape[-1] == 3:
                tensor = tensor.permute(2, 0, 1)
            return tensor.to(self.device).float().contiguous()
        return torch.from_numpy(frame_hr.transpose(2, 0, 1).copy()).to(self.device).float()

    def run(self, frame_hr: np.ndarray | torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tensor = self._to_tensor(frame_hr)
        _, boxes, scores, labels = self.detector.inference_raw(tensor)
        return boxes, scores, labels

    def run_batch(
        self,
        frames_hr: list[np.ndarray | torch.Tensor],
    ) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], list[int], dict[str, float]]:
        """Run detector in chunks that respect the engine max batch size."""

        if not frames_hr:
            return [], [], {}

        convert_started = time.time()
        with nvtx_range(f"detect:to_tensor n={len(frames_hr)}"):
            tensors = [self._to_tensor(frame_hr) for frame_hr in frames_hr]
        convert_seconds = time.time() - convert_started
        supports_batch = hasattr(self.detector, "inference_raw_batch")
        if not supports_batch or self.max_batch <= 1:
            infer_started = time.time()
            results = [self.run(tensor) for tensor in tensors]
            return results, [1] * len(results), {
                "to_tensor_seconds": convert_seconds,
                "preprocess_seconds": 0.0,
                "trt_seconds": time.time() - infer_started,
                "parse_seconds": 0.0,
            }

        chunk_size = max(1, self.max_batch)
        results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        plan: list[int] = []
        preprocess_seconds = 0.0
        trt_seconds = 0.0
        trt_setup_seconds = 0.0
        trt_enqueue_seconds = 0.0
        trt_sync_seconds = 0.0
        parse_seconds = 0.0
        offset = 0
        while offset < len(tensors):
            current = min(chunk_size, len(tensors) - offset)
            chunk_idx = offset // chunk_size
            with nvtx_range(f"detect:stack chunk={chunk_idx} batch={current}"):
                batch_tensor = torch.stack(tensors[offset:offset + current], dim=0).contiguous()
            with nvtx_range(f"detect:engine chunk={chunk_idx} batch={current}"):
                batch_results = self.detector.inference_raw_batch(batch_tensor)
            detector_timing = getattr(self.detector, "last_timing", {}) or {}
            preprocess_seconds += float(detector_timing.get("preprocess_seconds", 0.0))
            trt_seconds += float(detector_timing.get("trt_seconds", 0.0))
            trt_setup_seconds += float(detector_timing.get("trt_setup_seconds", 0.0))
            trt_enqueue_seconds += float(detector_timing.get("trt_enqueue_seconds", 0.0))
            trt_sync_seconds += float(detector_timing.get("trt_sync_seconds", 0.0))
            parse_seconds += float(detector_timing.get("parse_seconds", 0.0))
            if len(batch_results) != current:
                raise RuntimeError(
                    f"Detector returned {len(batch_results)} results for batch size {current}"
                )
            results.extend(batch_results)
            plan.append(current)
            offset += current
        return results, plan, {
            "to_tensor_seconds": convert_seconds,
            "preprocess_seconds": preprocess_seconds,
            "trt_seconds": trt_seconds,
            "trt_setup_seconds": trt_setup_seconds,
            "trt_enqueue_seconds": trt_enqueue_seconds,
            "trt_sync_seconds": trt_sync_seconds,
            "parse_seconds": parse_seconds,
        }


class GOPOrchestrator:
    """Coordinates GOP proposal, CPU finalization, SR, blend, and detection."""

    def __init__(
        self,
        *,
        video_dir: str,
        matcher: ProposalMatcher,
        sr_model: Any,
        detector: Any,
        output_dir: str,
        proposal_generator: OnlineProposalGenerator,
        proposal_cache_path: str | None = None,
        gop: int = 30,
        num_bins: int | None = None,
        max_sr_bins: int | None = 8,
        sr_batch_latency_ms: dict[int, float] | None = None,
        sr_launch_policy: str = "exact",
        save_outputs: bool = True,
        detection_device: str | torch.device | None = None,
        sr_device: str | torch.device | None = None,
        cross_camera_dedup: bool = True,
    ):
        self.video_dir = video_dir
        self.proposal_generator = proposal_generator
        self.proposal_cache_path = proposal_cache_path
        self.matcher = matcher
        self.sr_model = sr_model
        self.detector_runner = DetectionRunner(detector, device=detection_device)
        self.output_dir = output_dir
        self.gop = gop
        # Kept only for backward compatibility with older CLI/configs. Packing
        # now opens bins lazily instead of pre-allocating a fixed batch size.
        self.num_bins = num_bins
        self.max_sr_bins = max_sr_bins
        self.sr_batch_latency_ms = (
            {int(batch_size): float(latency) for batch_size, latency in sr_batch_latency_ms.items()}
            if sr_batch_latency_ms else None
        )
        if sr_launch_policy not in {"exact", "profile"}:
            raise ValueError(f"Unsupported SR launch policy: {sr_launch_policy}")
        self.sr_launch_policy = sr_launch_policy
        self.save_outputs = bool(save_outputs)
        self.sr_device = sr_device
        self.cross_camera_dedup = bool(cross_camera_dedup)
        if proposal_generator is not None and proposal_generator.camera_ids:
            self.cameras = proposal_generator.camera_ids
        else:
            self.cameras = homography_loader_cameras(matcher.homography_loader)
        self._proposal_pool = (
            ProposalWorkerPool(
                self.cameras,
                video_dir=self.video_dir,
                cfg=self.proposal_generator.cfg,
                gop=self.gop,
            )
            if self.proposal_generator is not None
            else None
        )
        self._caps = None if self._proposal_pool is not None else self._open_captures()
        self._gop_counter = 0
        self.runtime_stats: list[dict[str, Any]] = []
        self.detection_records: list[dict[str, Any]] = []

        if self.save_outputs:
            Path(os.path.join(output_dir, "visualizations")).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(output_dir, "sr_bins")).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _single_view_clusters(frame_id: int, frame_props: dict[str, list[Proposal]]) -> list[Cluster]:
        clusters: list[Cluster] = []
        cluster_idx = 0
        for cam_id in sorted(frame_props):
            for proposal in frame_props.get(cam_id, []):
                clusters.append(
                    Cluster(
                        cluster_id=f"f{frame_id:06d}_{cam_id}_p{cluster_idx:04d}",
                        frame_id=frame_id,
                        proposals=[proposal],
                        metadata={"baseline": "object_no_cross_camera_dedup"},
                    )
                )
                cluster_idx += 1
        return clusters

    def _open_captures(self) -> dict[str, cv2.VideoCapture]:
        caps = {}
        for cam in self.cameras:
            path = os.path.join(self.video_dir, f"{cam}_aligned.avi")
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video for {cam}: {path}")
            caps[cam] = cap
        return caps

    def seek(self, start_frame: int) -> None:
        if self._caps is None:
            return
        for cap in self._caps.values():
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame - 1))

    def close(self) -> None:
        if self._proposal_pool is not None:
            self._proposal_pool.close()
        if self._caps is not None:
            for cap in self._caps.values():
                cap.release()

    def _decode_gop(
        self,
        start_frame: int,
        max_frame: int | None = None,
    ) -> tuple[list[int], dict[int, dict[str, np.ndarray]]]:
        frame_ids = []
        frames_by_frame: dict[int, dict[str, np.ndarray]] = {}
        for offset in range(self.gop):
            frame_id = start_frame + offset
            if max_frame is not None and frame_id > max_frame:
                break
            frames = {}
            for cam, cap in self._caps.items():
                ret, frame = cap.read()
                if ret:
                    frames[cam] = frame
            if not frames:
                break
            frame_ids.append(frame_id)
            frames_by_frame[frame_id] = frames
        return frame_ids, frames_by_frame

    def _submit_worker_gop(
        self,
        start_frame: int,
        max_frame: int | None,
        gop_index: int,
    ) -> WorkerGOPJob:
        if self._proposal_pool is None:
            raise RuntimeError("Worker GOP submission requires ProposalWorkerPool")
        job_id = f"g{gop_index:06d}_f{start_frame:06d}"
        submitted_at = time.time()
        self._proposal_pool.submit(job_id, start_frame=start_frame, max_frame=max_frame)
        return WorkerGOPJob(
            job_id=job_id,
            gop_index=int(gop_index),
            start_frame=int(start_frame),
            max_frame=max_frame,
            submitted_at=submitted_at,
        )

    def _collect_worker_gop(self, job: WorkerGOPJob) -> RawGOPInput | None:
        """Turn worker payloads into shared-memory frame views and proposals."""

        if self._proposal_pool is None:
            raise RuntimeError("Worker GOP collection requires ProposalWorkerPool")
        worker_results = self._proposal_pool.collect(job.job_id)
        _ = time.time() - job.submitted_at

        frame_ids = sorted(
            {
                int(frame_id)
                for payload in worker_results.values()
                for frame_id in payload.get("frame_ids", [])
            }
        )
        if not frame_ids:
            return None
        active_frame_ids = sorted(
            {
                int(frame_id)
                for payload in worker_results.values()
                for frame_id in payload.get("recorded_frame_ids", [])
            }
        )

        frames_by_frame: dict[int, dict[str, np.ndarray]] = {frame_id: {} for frame_id in frame_ids}
        frame_refs_by_frame: dict[int, dict[str, dict[str, int]]] = {frame_id: {} for frame_id in frame_ids}
        proposals: dict[int, dict[str, list[Proposal]]] = {}
        worker_stats_by_camera: dict[str, dict[str, float]] = {}
        raw_proposal_count = 0
        raw_proposal_area = 0
        decode_time = 0.0
        proposal_bgsub_time = 0.0
        proposal_morphology_time = 0.0
        proposal_contour_time = 0.0
        proposal_consolidate_time = 0.0
        proposal_split_time = 0.0
        proposal_materialize_time = 0.0

        for cam_id in self.cameras:
            payload = worker_results.get(cam_id)
            if payload is None:
                continue
            stats = payload.get("stats", {})
            worker_stats_by_camera[cam_id] = {
                "worker_total_seconds": float(stats.get("worker_total_seconds", 0.0)),
                "worker_total_with_put_seconds": float(stats.get("worker_total_with_put_seconds", 0.0)),
                "decode_seconds": float(stats.get("decode_seconds", 0.0)),
                "shm_write_seconds": float(stats.get("shm_write_seconds", 0.0)),
                "proposal_process_seconds": float(stats.get("proposal_process_seconds", 0.0)),
                "queue_put_seconds": float(stats.get("queue_put_seconds", 0.0)),
                "queue_get_seconds": float(stats.get("queue_get_seconds", 0.0)),
                "queue_put_stats_get_seconds": float(stats.get("queue_put_stats_get_seconds", 0.0)),
                "collect_total_seconds": float(stats.get("collect_total_seconds", 0.0)),
                "frame_bytes": float(stats.get("frame_bytes", 0.0)),
                "bgsub_seconds": float(stats.get("bgsub_seconds", 0.0)),
                "gray_seconds": float(stats.get("gray_seconds", 0.0)),
                "mog2_seconds": float(stats.get("mog2_seconds", 0.0)),
                "morphology_seconds": float(stats.get("morphology_seconds", 0.0)),
                "contour_seconds": float(stats.get("contour_seconds", 0.0)),
                "consolidate_seconds": float(stats.get("consolidate_seconds", 0.0)),
                "split_seconds": float(stats.get("split_seconds", 0.0)),
                "materialize_seconds": float(stats.get("materialize_seconds", 0.0)),
            }
            decode_time = max(decode_time, float(stats.get("decode_seconds", 0.0)))
            proposal_bgsub_time += float(stats.get("bgsub_seconds", 0.0))
            proposal_morphology_time += float(stats.get("morphology_seconds", 0.0))
            proposal_contour_time += float(stats.get("contour_seconds", 0.0))
            proposal_consolidate_time += float(stats.get("consolidate_seconds", 0.0))
            proposal_split_time += float(stats.get("split_seconds", 0.0))
            proposal_materialize_time += float(stats.get("materialize_seconds", 0.0))

            for frame_id, frame_ref in payload.get("frame_refs_by_frame", {}).items():
                frame_id = int(frame_id)
                frames_by_frame.setdefault(frame_id, {})[cam_id] = self._proposal_pool.frame_view(
                    cam_id,
                    int(frame_ref["slot"]),
                    int(frame_ref["offset"]),
                )
                frame_refs_by_frame.setdefault(frame_id, {})[cam_id] = {
                    "slot": int(frame_ref["slot"]),
                    "offset": int(frame_ref["offset"]),
                }
            for frame_id, cam_props in payload.get("proposals_by_frame", {}).items():
                frame_id = int(frame_id)
                proposals.setdefault(frame_id, {})[cam_id] = cam_props
                raw_proposal_count += len(cam_props)
                raw_proposal_area += sum(_bbox_area(proposal.bbox) for proposal in cam_props)
                if self.proposal_generator is not None:
                    self.proposal_generator.generated.extend(cam_props)

        worker_max_total = max(
            (stats.get("worker_total_seconds", 0.0) for stats in worker_stats_by_camera.values()),
            default=0.0,
        )
        proposal_time = max(0.0, worker_max_total - decode_time)
        proposal_sync_overhead = 0.0
        collect_seconds = max(
            (stats.get("collect_total_seconds", 0.0) for stats in worker_stats_by_camera.values()),
            default=0.0,
        )
        for frame_id in frame_ids:
            proposals.setdefault(frame_id, {})
            frames_by_frame.setdefault(frame_id, {})

        return RawGOPInput(
            gop_index=job.gop_index,
            frame_ids=frame_ids,
            active_frame_ids=active_frame_ids,
            frames=frames_by_frame,
            frame_refs=frame_refs_by_frame,
            proposals=proposals,
            raw_proposal_count=raw_proposal_count,
            raw_proposal_area=raw_proposal_area,
            decode_seconds=decode_time,
            proposal_seconds=proposal_time,
            proposal_bgsub_seconds=proposal_bgsub_time,
            proposal_morphology_seconds=proposal_morphology_time,
            proposal_contour_seconds=proposal_contour_time,
            proposal_consolidate_seconds=proposal_consolidate_time,
            proposal_split_seconds=proposal_split_time,
            proposal_materialize_seconds=proposal_materialize_time,
            worker_stats=worker_stats_by_camera,
            collect_seconds=collect_seconds,
            proposal_sync_overhead_seconds=proposal_sync_overhead,
        )

    def _finalize_raw_gop(self, raw: RawGOPInput, cpu_started: float) -> PreparedGOP:
        """Run main-thread CPU work after proposal collection.

        This stage performs cross-camera matching, ranks the best view from each
        cluster, applies the SR-bin budget, and computes bin placements. The
        actual image crops are still deferred to GPU materialization.
        """

        clusters_by_frame: dict[int, list[Cluster]] = {}
        candidates: list[CandidatePatch] = []
        match_time = 0.0
        ranking_time = 0.0

        for frame_id in raw.frame_ids:
            frame_props = raw.proposals.get(frame_id, {})
            if self.cross_camera_dedup:
                t0 = time.time()
                clusters = self.matcher.match_frame(frame_id, frame_props)
                match_time += time.time() - t0
            else:
                clusters = self._single_view_clusters(frame_id, frame_props)
            clusters_by_frame[frame_id] = clusters

            t0 = time.time()
            for cluster in clusters:
                obj_views = {proposal.camera_id: proposal.bbox for proposal in cluster.proposals}
                frame_map = raw.frames[frame_id]
                debug_label = None
                if raw.gop_index == 31:
                    debug_label = f"gop={raw.gop_index} frame={frame_id} cluster={cluster.cluster_id}"
                q_scores = compute_view_quality(frame_map, obj_views, debug_label=debug_label)
                if not q_scores:
                    continue
                best_cam = select_best_view(q_scores)
                frame = frame_map.get(best_cam)
                if frame is None:
                    continue
                bbox = obj_views[best_cam]
                _, _, bw, bh = bbox
                if bw <= 0 or bh <= 0:
                    continue
                importance = compute_importance(obj_views, q_scores, best_cam, debug_label=debug_label)
                candidates.append(
                    CandidatePatch(
                        cluster_id=cluster.cluster_id,
                        cam_id=best_cam,
                        frame_id=frame_id,
                        patch=None,
                        # Keep only bbox/shape metadata here. The GPU later
                        # crops from the full copied frame, avoiding per-patch
                        # CPU materialization and H2D copies.
                        orig_bbox=bbox,
                        importance=importance,
                        metadata={
                            "member_cameras": [proposal.camera_id for proposal in cluster.proposals],
                            "proposal_ids": [proposal.proposal_id for proposal in cluster.proposals],
                            "q_scores": q_scores,
                        },
                        patch_w=int(bw),
                        patch_h=int(bh),
                    )
                )
            ranking_time += time.time() - t0

        t0 = time.time()
        normalize_candidate_importance(candidates)
        packing_profile: dict[str, float | int] = {}
        selected_candidates, budget_deferred = select_candidates_under_bin_budget(
            candidates,
            self.max_sr_bins,
            profile=packing_profile,
        )
        bins, placements, invalid_deferred = pack_objects(selected_candidates, profile=packing_profile)
        deferred = budget_deferred + invalid_deferred
        packing_time = time.time() - t0
        _ = cpu_started
        finalize_seconds = match_time + ranking_time + packing_time
        cpu_stage_total = (
            raw.decode_seconds
            + raw.proposal_seconds
            + finalize_seconds
        )
        cpu_total = raw.collect_seconds + finalize_seconds

        dedup_candidate_area = sum(_bbox_area(candidate.orig_bbox) for candidate in candidates)
        placed_candidate_area = sum(_bbox_area(placement.orig_bbox) for placement in placements)
        deferred_candidate_area = sum(_bbox_area(candidate.orig_bbox) for candidate in deferred)
        sr_canvas_pixels = len(bins) * int(race_core.BIN_W) * int(race_core.BIN_H)
        area_stats = {
            "cross_camera_dedup": self.cross_camera_dedup,
            "raw_proposal_count": raw.raw_proposal_count,
            "raw_proposal_area": raw.raw_proposal_area,
            "dedup_candidate_count": len(candidates),
            "dedup_candidate_area": dedup_candidate_area,
            "placed_candidate_count": len(placements),
            "placed_candidate_area": placed_candidate_area,
            "max_sr_bins": self.max_sr_bins,
            "budget_selected_candidate_count": len(selected_candidates),
            "budget_deferred_candidate_count": len(budget_deferred),
            "deferred_candidate_count": len(deferred),
            "deferred_candidate_area": deferred_candidate_area,
            "opened_bin_count": len(bins),
            "sr_canvas_pixels": sr_canvas_pixels,
            "placed_pixels": placed_candidate_area,
            "fill_ratio": (float(placed_candidate_area) / float(sr_canvas_pixels)) if sr_canvas_pixels > 0 else 0.0,
            "raw_bins_640x360": _estimate_bin_count(raw.raw_proposal_area, RAW_BIN_AREA_640_360),
            "raw_bins_256x256": _estimate_bin_count(raw.raw_proposal_area, RAW_BIN_AREA_256_256),
            "dedup_bins_640x360": _estimate_bin_count(dedup_candidate_area, RAW_BIN_AREA_640_360),
            "dedup_bins_256x256": _estimate_bin_count(dedup_candidate_area, RAW_BIN_AREA_256_256),
        }

        return PreparedGOP(
            gop_index=raw.gop_index,
            frame_ids=raw.frame_ids,
            active_frame_ids=raw.active_frame_ids,
            frames=raw.frames,
            frame_refs=raw.frame_refs,
            proposals=raw.proposals,
            clusters=clusters_by_frame,
            candidates=candidates,
            bins=bins,
            placements=placements,
            deferred=deferred,
            cpu_stats={
                "decode_seconds": raw.decode_seconds,
                "proposal_seconds": raw.proposal_seconds,
                "proposal_bgsub_seconds": raw.proposal_bgsub_seconds,
                "proposal_morphology_seconds": raw.proposal_morphology_seconds,
                "proposal_contour_seconds": raw.proposal_contour_seconds,
                "proposal_consolidate_seconds": raw.proposal_consolidate_seconds,
                "proposal_split_seconds": raw.proposal_split_seconds,
                "proposal_materialize_seconds": raw.proposal_materialize_seconds,
                "proposal_sync_overhead_seconds": raw.proposal_sync_overhead_seconds,
                "proposal_collect_seconds": raw.collect_seconds,
                "match_seconds": match_time,
                "ranking_seconds": ranking_time,
                "packing_seconds": packing_time,
                "finalize_seconds": finalize_seconds,
                "cpu_stage_seconds": cpu_stage_total,
                "worker_hidden_seconds": max(0.0, cpu_stage_total - cpu_total),
                "candidate_sort_seconds": float(packing_profile.get("candidate_sort_seconds", 0.0)),
                "admission_seconds": float(packing_profile.get("admission_seconds", 0.0)),
                "compact_pack_seconds": float(packing_profile.get("compact_pack_seconds", 0.0)),
                "render_bin_seconds": float(packing_profile.get("render_bin_seconds", 0.0)),
                "repack_count": int(packing_profile.get("repack_count", 0)),
                "free_rect_count": int(packing_profile.get("free_rect_count", 0)),
                "cpu_total_seconds": cpu_total,
            },
            area_stats=area_stats,
            worker_stats=raw.worker_stats,
        )

    def prepare_gop(self, start_frame: int, max_frame: int | None = None) -> PreparedGOP | None:
        cpu_started = time.time()

        proposal_time = 0.0
        decode_time = 0.0
        proposal_bgsub_time = 0.0
        proposal_morphology_time = 0.0
        proposal_contour_time = 0.0
        proposal_consolidate_time = 0.0
        proposal_split_time = 0.0
        proposal_materialize_time = 0.0
        proposals: dict[int, dict[str, list[Proposal]]] = {}
        frame_refs_by_frame: dict[int, dict[str, dict[str, int]]] = {}
        raw_proposal_count = 0
        raw_proposal_area = 0
        active_frame_ids: list[int] = []
        worker_stats_by_camera: dict[str, dict[str, float]] = {}
        if self._proposal_pool is not None:
            proposal_started = time.time()
            job_id = f"g{self._gop_counter:06d}_f{start_frame:06d}"
            self._proposal_pool.submit(job_id, start_frame=start_frame, max_frame=max_frame)
            worker_results = self._proposal_pool.collect(job_id)
            proposal_phase_wall = time.time() - proposal_started

            frame_ids = sorted(
                {
                    int(frame_id)
                    for payload in worker_results.values()
                    for frame_id in payload.get("frame_ids", [])
                }
            )
            if not frame_ids:
                return None
            active_frame_ids = sorted(
                {
                    int(frame_id)
                    for payload in worker_results.values()
                    for frame_id in payload.get("recorded_frame_ids", [])
                }
            )

            frames_by_frame: dict[int, dict[str, np.ndarray]] = {frame_id: {} for frame_id in frame_ids}
            for cam_id in self.cameras:
                payload = worker_results.get(cam_id)
                if payload is None:
                    continue
                stats = payload.get("stats", {})
                worker_stats_by_camera[cam_id] = {
                    "worker_total_seconds": float(stats.get("worker_total_seconds", 0.0)),
                    "worker_total_with_put_seconds": float(stats.get("worker_total_with_put_seconds", 0.0)),
                    "decode_seconds": float(stats.get("decode_seconds", 0.0)),
                    "shm_write_seconds": float(stats.get("shm_write_seconds", 0.0)),
                    "proposal_process_seconds": float(stats.get("proposal_process_seconds", 0.0)),
                    "queue_put_seconds": float(stats.get("queue_put_seconds", 0.0)),
                    "queue_get_seconds": float(stats.get("queue_get_seconds", 0.0)),
                    "queue_put_stats_get_seconds": float(stats.get("queue_put_stats_get_seconds", 0.0)),
                    "collect_total_seconds": float(stats.get("collect_total_seconds", 0.0)),
                    "frame_bytes": float(stats.get("frame_bytes", 0.0)),
                    "bgsub_seconds": float(stats.get("bgsub_seconds", 0.0)),
                    "gray_seconds": float(stats.get("gray_seconds", 0.0)),
                    "mog2_seconds": float(stats.get("mog2_seconds", 0.0)),
                    "morphology_seconds": float(stats.get("morphology_seconds", 0.0)),
                    "contour_seconds": float(stats.get("contour_seconds", 0.0)),
                    "consolidate_seconds": float(stats.get("consolidate_seconds", 0.0)),
                    "split_seconds": float(stats.get("split_seconds", 0.0)),
                    "materialize_seconds": float(stats.get("materialize_seconds", 0.0)),
                }
                decode_time = max(decode_time, float(stats.get("decode_seconds", 0.0)))
                proposal_bgsub_time += float(stats.get("bgsub_seconds", 0.0))
                proposal_morphology_time += float(stats.get("morphology_seconds", 0.0))
                proposal_contour_time += float(stats.get("contour_seconds", 0.0))
                proposal_consolidate_time += float(stats.get("consolidate_seconds", 0.0))
                proposal_split_time += float(stats.get("split_seconds", 0.0))
                proposal_materialize_time += float(stats.get("materialize_seconds", 0.0))

                for frame_id, frame_ref in payload.get("frame_refs_by_frame", {}).items():
                    frame_id = int(frame_id)
                    frames_by_frame.setdefault(frame_id, {})[cam_id] = self._proposal_pool.frame_view(
                        cam_id,
                        int(frame_ref["slot"]),
                        int(frame_ref["offset"]),
                    )
                    frame_refs_by_frame.setdefault(frame_id, {})[cam_id] = {
                        "slot": int(frame_ref["slot"]),
                        "offset": int(frame_ref["offset"]),
                    }
                for frame_id, cam_props in payload.get("proposals_by_frame", {}).items():
                    frame_id = int(frame_id)
                    proposals.setdefault(frame_id, {})[cam_id] = cam_props
                    raw_proposal_count += len(cam_props)
                    raw_proposal_area += sum(_bbox_area(proposal.bbox) for proposal in cam_props)
                    if self.proposal_generator is not None:
                        self.proposal_generator.generated.extend(cam_props)

            proposal_time = max(0.0, proposal_phase_wall - decode_time)
        else:
            frame_ids, frames_by_frame = self._decode_gop(start_frame, max_frame=max_frame)
            if not frame_ids:
                return None

            for frame_id in frame_ids:
                t0 = time.time()
                frame_props, frame_stats = self.proposal_generator.process_frame(frame_id, frames_by_frame[frame_id])
                if bool(frame_stats.get("record", False)):
                    active_frame_ids.append(frame_id)
                proposal_time += time.time() - t0
                proposal_bgsub_time += float(frame_stats.get("bgsub_seconds", 0.0))
                proposal_morphology_time += float(frame_stats.get("morphology_seconds", 0.0))
                proposal_contour_time += float(frame_stats.get("contour_seconds", 0.0))
                proposal_consolidate_time += float(frame_stats.get("consolidate_seconds", 0.0))
                proposal_split_time += float(frame_stats.get("split_seconds", 0.0))
                proposal_materialize_time += float(frame_stats.get("materialize_seconds", 0.0))
                proposals[frame_id] = frame_props
                for cam_props in frame_props.values():
                    raw_proposal_count += len(cam_props)
                    raw_proposal_area += sum(_bbox_area(proposal.bbox) for proposal in cam_props)
            for cam_id in self.cameras:
                worker_stats_by_camera[cam_id] = {
                    "worker_total_seconds": 0.0,
                    "worker_total_with_put_seconds": 0.0,
                    "decode_seconds": 0.0,
                    "shm_write_seconds": 0.0,
                    "proposal_process_seconds": 0.0,
                    "queue_put_seconds": 0.0,
                    "queue_get_seconds": 0.0,
                    "queue_put_stats_get_seconds": 0.0,
                    "collect_total_seconds": 0.0,
                    "frame_bytes": 0.0,
                    "bgsub_seconds": 0.0,
                    "gray_seconds": 0.0,
                    "mog2_seconds": 0.0,
                    "morphology_seconds": 0.0,
                    "contour_seconds": 0.0,
                    "consolidate_seconds": 0.0,
                    "split_seconds": 0.0,
                    "materialize_seconds": 0.0,
                }

        for frame_id in frame_ids:
            proposals.setdefault(frame_id, {})
            frames_by_frame.setdefault(frame_id, {})
            frame_refs_by_frame.setdefault(frame_id, {})
        clusters_by_frame: dict[int, list[Cluster]] = {}
        candidates: list[CandidatePatch] = []
        match_time = 0.0
        ranking_time = 0.0

        for frame_id in frame_ids:
            frame_props = proposals.get(frame_id, {})
            if self.cross_camera_dedup:
                t0 = time.time()
                clusters = self.matcher.match_frame(frame_id, frame_props)
                match_time += time.time() - t0
            else:
                clusters = self._single_view_clusters(frame_id, frame_props)
            clusters_by_frame[frame_id] = clusters

            t0 = time.time()
            for cluster in clusters:
                obj_views = {proposal.camera_id: proposal.bbox for proposal in cluster.proposals}
                frame_map = frames_by_frame[frame_id]
                debug_label = None
                if self._gop_counter == 31:
                    debug_label = f"gop={self._gop_counter} frame={frame_id} cluster={cluster.cluster_id}"
                q_scores = compute_view_quality(frame_map, obj_views, debug_label=debug_label)
                if not q_scores:
                    continue
                best_cam = select_best_view(q_scores)
                frame = frame_map.get(best_cam)
                if frame is None:
                    continue
                bbox = obj_views[best_cam]
                _, _, bw, bh = bbox
                if bw <= 0 or bh <= 0:
                    continue
                importance = compute_importance(obj_views, q_scores, best_cam, debug_label=debug_label)
                candidates.append(
                    CandidatePatch(
                        cluster_id=cluster.cluster_id,
                        cam_id=best_cam,
                        frame_id=frame_id,
                        patch=None,
                        orig_bbox=bbox,
                        importance=importance,
                        metadata={
                            "member_cameras": [proposal.camera_id for proposal in cluster.proposals],
                            "proposal_ids": [proposal.proposal_id for proposal in cluster.proposals],
                            "q_scores": q_scores,
                        },
                        patch_w=int(bw),
                        patch_h=int(bh),
                    )
                )
            ranking_time += time.time() - t0

        t0 = time.time()
        normalize_candidate_importance(candidates)
        packing_profile: dict[str, float | int] = {}
        selected_candidates, budget_deferred = select_candidates_under_bin_budget(
            candidates,
            self.max_sr_bins,
            profile=packing_profile,
        )
        bins, placements, invalid_deferred = pack_objects(selected_candidates, profile=packing_profile)
        # if self._gop_counter == 31:
        #     for candidate in candidates:
        #         print(
        #             "[importance_norm] "
        #             f"gop={self._gop_counter} "
        #             f"frame={candidate.frame_id} "
        #             f"cluster={candidate.cluster_id} "
        #             f"cam={candidate.cam_id} "
        #             f"raw={raw_importance_by_cluster.get(candidate.cluster_id, 0.0):.6f} "
        #             f"norm={candidate.importance:.6f}",
        #             flush=True,
        #         )
        #     print(
        #         "[packing_budget] "
        #         f"gop={self._gop_counter} "
        #         f"max_sr_bins={self.max_sr_bins} "
        #         f"candidates={len(candidates)} "
        #         f"selected={len(selected_candidates)} "
        #         f"budget_deferred={len(budget_deferred)}",
        #         flush=True,
        #     )
        deferred = budget_deferred + invalid_deferred
        packing_time = time.time() - t0
        cpu_total = time.time() - cpu_started
        proposal_sync_overhead = 0.0
        if self._proposal_pool is None:
            decode_time = max(0.0, cpu_total - proposal_time - match_time - ranking_time - packing_time)
        else:
            worker_max_total = max(
                (stats.get("worker_total_seconds", 0.0) for stats in worker_stats_by_camera.values()),
                default=0.0,
            )
            proposal_sync_overhead = max(0.0, proposal_time - worker_max_total)

        dedup_candidate_area = sum(_bbox_area(candidate.orig_bbox) for candidate in candidates)
        placed_candidate_area = sum(_bbox_area(placement.orig_bbox) for placement in placements)
        deferred_candidate_area = sum(_bbox_area(candidate.orig_bbox) for candidate in deferred)
        sr_canvas_pixels = len(bins) * int(race_core.BIN_W) * int(race_core.BIN_H)
        area_stats = {
            "cross_camera_dedup": self.cross_camera_dedup,
            "raw_proposal_count": raw_proposal_count,
            "raw_proposal_area": raw_proposal_area,
            "dedup_candidate_count": len(candidates),
            "dedup_candidate_area": dedup_candidate_area,
            "placed_candidate_count": len(placements),
            "placed_candidate_area": placed_candidate_area,
            "max_sr_bins": self.max_sr_bins,
            "budget_selected_candidate_count": len(selected_candidates),
            "budget_deferred_candidate_count": len(budget_deferred),
            "deferred_candidate_count": len(deferred),
            "deferred_candidate_area": deferred_candidate_area,
            "opened_bin_count": len(bins),
            "sr_canvas_pixels": sr_canvas_pixels,
            "placed_pixels": placed_candidate_area,
            "fill_ratio": (float(placed_candidate_area) / float(sr_canvas_pixels)) if sr_canvas_pixels > 0 else 0.0,
            "raw_bins_640x360": _estimate_bin_count(raw_proposal_area, RAW_BIN_AREA_640_360),
            "raw_bins_256x256": _estimate_bin_count(raw_proposal_area, RAW_BIN_AREA_256_256),
            "dedup_bins_640x360": _estimate_bin_count(dedup_candidate_area, RAW_BIN_AREA_640_360),
            "dedup_bins_256x256": _estimate_bin_count(dedup_candidate_area, RAW_BIN_AREA_256_256),
        }

        return PreparedGOP(
            gop_index=self._gop_counter,
            frame_ids=frame_ids,
            active_frame_ids=active_frame_ids,
            frames=frames_by_frame,
            frame_refs=frame_refs_by_frame,
            proposals=proposals,
            clusters=clusters_by_frame,
            candidates=candidates,
            bins=bins,
            placements=placements,
            deferred=deferred,
            cpu_stats={
                "decode_seconds": decode_time,
                "proposal_seconds": proposal_time,
                "proposal_bgsub_seconds": proposal_bgsub_time,
                "proposal_morphology_seconds": proposal_morphology_time,
                "proposal_contour_seconds": proposal_contour_time,
                "proposal_consolidate_seconds": proposal_consolidate_time,
                "proposal_split_seconds": proposal_split_time,
                "proposal_materialize_seconds": proposal_materialize_time,
                "proposal_sync_overhead_seconds": proposal_sync_overhead,
                "match_seconds": match_time,
                "ranking_seconds": ranking_time,
                "packing_seconds": packing_time,
                "candidate_sort_seconds": float(packing_profile.get("candidate_sort_seconds", 0.0)),
                "admission_seconds": float(packing_profile.get("admission_seconds", 0.0)),
                "compact_pack_seconds": float(packing_profile.get("compact_pack_seconds", 0.0)),
                "render_bin_seconds": float(packing_profile.get("render_bin_seconds", 0.0)),
                "repack_count": int(packing_profile.get("repack_count", 0)),
                "free_rect_count": int(packing_profile.get("free_rect_count", 0)),
                "cpu_total_seconds": cpu_total,
            },
            area_stats=area_stats,
            worker_stats=worker_stats_by_camera,
        )

    def _save_sr_bins(
        self,
        prepared: PreparedGOP,
        sr_bins: list[np.ndarray | torch.Tensor],
        *,
        sr_inputs: list[np.ndarray | torch.Tensor] | None = None,
    ) -> None:
        if not self.save_outputs:
            return
        for idx, current_bin in enumerate(prepared.bins):
            base = os.path.join(self.output_dir, "sr_bins", f"g{prepared.gop_index:04d}_bin{idx:02d}")
            input_image = current_bin.image
            if sr_inputs is not None and idx < len(sr_inputs):
                input_image = tensor_to_bgr_image(sr_inputs[idx]) if torch.is_tensor(sr_inputs[idx]) else sr_inputs[idx]
            cv2.imwrite(base + "_input.jpg", input_image)
            if idx < len(sr_bins):
                sr_image = tensor_to_bgr_image(sr_bins[idx]) if torch.is_tensor(sr_bins[idx]) else sr_bins[idx]
                cv2.imwrite(base + "_sr.jpg", sr_image)

    def _save_frame_result(
        self,
        frame_id: int,
        cam_id: str,
        frame_hr: np.ndarray | torch.Tensor,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        if not self.save_outputs:
            return
        if torch.is_tensor(frame_hr):
            vis = tensor_to_bgr_image(frame_hr)
        else:
            vis = frame_hr.copy()
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                vis,
                f"{int(label)}:{float(score):.2f}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        out_path = os.path.join(self.output_dir, "visualizations", f"frame{frame_id:06d}_{cam_id}.jpg")
        cv2.imwrite(out_path, vis)

    def process_gop(self, prepared: PreparedGOP) -> dict[str, Any]:
        """Execute the serialized main GPU path for one prepared GOP.

        Data flow: shared-memory frame views -> CUDA frame batch -> GPU SR bins
        -> SR engine -> blend all HR frames -> detector chunks -> detections.
        """

        gpu_started = time.time()
        with nvtx_range(f"gop:{prepared.gop_index:04d}:stage:gpu_total"):
            detect_targets: list[tuple[int, str, np.ndarray, dict[str, int] | None]] = []
            for cam_id in self.cameras:
                for frame_id in prepared.active_frame_ids:
                    frame_lr = prepared.frames[frame_id].get(cam_id)
                    if frame_lr is None:
                        continue
                    detect_targets.append((frame_id, cam_id, frame_lr, prepared.frame_refs.get(frame_id, {}).get(cam_id)))
            detect_frame_copy_started = time.time()
            detect_frame_cpu_stage_seconds = 0.0
            detect_frame_h2d_seconds = 0.0
            detect_frame_gpu_convert_seconds = 0.0
            with nvtx_range(f"gop:{prepared.gop_index:04d}:stage:frame_h2d n={len(detect_targets)}"):
                detect_frame_batch, detect_frame_timing = _copy_detect_targets_to_gpu(
                    detect_targets,
                    device=self.sr_device,
                    proposal_pool=self._proposal_pool,
                )
            detect_frame_cpu_stage_seconds = float(detect_frame_timing.get("cpu_stage_seconds", 0.0))
            detect_frame_h2d_seconds = float(detect_frame_timing.get("h2d_seconds", 0.0))
            detect_frame_gpu_convert_seconds = float(detect_frame_timing.get("gpu_convert_seconds", 0.0))
            detect_frame_copy_seconds = time.time() - detect_frame_copy_started

            sr_materialize_seconds = 0.0
            sr_materialize_bin_init_seconds = 0.0
            sr_materialize_patch_resize_seconds = 0.0
            sr_materialize_patch_paste_seconds = 0.0
            if self.sr_launch_policy == "profile":
                sr_launch_plan = optimize_launch_plan(len(prepared.bins), self.sr_batch_latency_ms)
            else:
                sr_launch_plan = [len(prepared.bins)] if prepared.bins else []
            sr_copy_seconds = 0.0
            sr_inference_seconds = 0.0
            sr_input_bins: list[torch.Tensor] = []
            sr_started = time.time()
            with nvtx_range(f"gop:{prepared.gop_index:04d}:stage:sr_total bins={len(prepared.bins)}"):
                if prepared.active_frame_ids and prepared.placements:
                    materialize_started = time.time()
                    with nvtx_range(f"gop:{prepared.gop_index:04d}:stage:bin_materialize n={len(prepared.bins)}"):
                        sr_input_bins, sr_materialize_timing = _materialize_bins_on_gpu(
                            detect_targets,
                            detect_frame_batch,
                            prepared.placements,
                            device=self.sr_device,
                            return_timing=True,
                        )
                    sr_materialize_seconds = time.time() - materialize_started
                    sr_materialize_bin_init_seconds = float(sr_materialize_timing.get("bin_init_seconds", 0.0))
                    sr_materialize_patch_resize_seconds = float(sr_materialize_timing.get("patch_resize_seconds", 0.0))
                    sr_materialize_patch_paste_seconds = float(sr_materialize_timing.get("patch_paste_seconds", 0.0))
                    with nvtx_range(
                        f"gop:{prepared.gop_index:04d}:stage:sr_infer bins={len(prepared.bins)} plan={sr_launch_plan}"
                    ):
                        sr_bins, sr_timing = run_sr_batch(
                            self.sr_model,
                            sr_input_bins,
                            device=self.sr_device,
                            launch_plan=sr_launch_plan,
                            return_tensors=True,
                            return_timing=True,
                        )
                        sr_copy_seconds = float(sr_timing.get("copy_seconds", 0.0))
                        sr_inference_seconds = float(sr_timing.get("inference_seconds", 0.0))
                else:
                    sr_bins = []
            sr_time = time.time() - sr_started
            with nvtx_range(f"gop:{prepared.gop_index:04d}:stage:save_sr_bins"):
                self._save_sr_bins(prepared, sr_bins, sr_inputs=sr_input_bins)

            det_started = time.time()
            placements_by_target: dict[tuple[int, str], list[PlacementEntry]] = defaultdict(list)
            for placement in prepared.placements:
                placements_by_target[(placement.frame_id, placement.cam_id)].append(placement)

            detect_batch_plan: list[int] = []
            detect_chunk_size = max(1, self.detector_runner.max_batch)
            detect_blend_seconds = 0.0
            detect_to_tensor_seconds = 0.0
            detect_preprocess_seconds = 0.0
            detect_trt_seconds = 0.0
            detect_trt_setup_seconds = 0.0
            detect_trt_enqueue_seconds = 0.0
            detect_trt_sync_seconds = 0.0
            detect_parse_seconds = 0.0
            detect_save_seconds = 0.0
        # Blend breakdown profiling is intentionally disabled on the hot path
        # because return_timing=True adds CUDA synchronizations.
        # detect_blend_to_tensor_seconds = 0.0
        # detect_blend_upsample_seconds = 0.0
        # detect_blend_patch_resize_seconds = 0.0
        # detect_blend_patch_paste_seconds = 0.0
            if detect_targets:
                blend_started = time.time()
                with nvtx_range(f"gop:{prepared.gop_index:04d}:stage:blend_all n={len(detect_targets)}"):
                    frame_hrs_all = blend_back_frames_torch(
                        detect_frame_batch,
                        sr_bins,
                        [placements_by_target.get((frame_id, cam_id), []) for frame_id, cam_id, _, _ in detect_targets],
                        device=self.sr_device,
                    )
                detect_blend_seconds += time.time() - blend_started
            else:
                frame_hrs_all = []

            with nvtx_range(f"gop:{prepared.gop_index:04d}:stage:detect_total inputs={len(detect_targets)}"):
                for offset in range(0, len(detect_targets), detect_chunk_size):
                    chunk_idx = offset // detect_chunk_size
                    chunk = detect_targets[offset:offset + detect_chunk_size]
                    frame_hrs = frame_hrs_all[offset:offset + len(chunk)]
                    with nvtx_range(
                        f"gop:{prepared.gop_index:04d}:stage:detect chunk={chunk_idx} n={len(frame_hrs)}"
                    ):
                        detection_results, chunk_plan, timing = self.detector_runner.run_batch(frame_hrs)
                    detect_batch_plan.extend(chunk_plan)
                    detect_to_tensor_seconds += float(timing.get("to_tensor_seconds", 0.0))
                    detect_preprocess_seconds += float(timing.get("preprocess_seconds", 0.0))
                    detect_trt_seconds += float(timing.get("trt_seconds", 0.0))
                    detect_trt_setup_seconds += float(timing.get("trt_setup_seconds", 0.0))
                    detect_trt_enqueue_seconds += float(timing.get("trt_enqueue_seconds", 0.0))
                    detect_trt_sync_seconds += float(timing.get("trt_sync_seconds", 0.0))
                    detect_parse_seconds += float(timing.get("parse_seconds", 0.0))
                    save_started = time.time()
                    with nvtx_range(
                        f"gop:{prepared.gop_index:04d}:stage:record_results chunk={chunk_idx}"
                    ):
                        for (frame_id, cam_id, _, _), frame_hr, (boxes, scores, labels) in zip(chunk, frame_hrs, detection_results):
                            self._save_frame_result(frame_id, cam_id, frame_hr, boxes, scores, labels)
                            self.detection_records.append(
                                {
                                    "frame_id": frame_id,
                                    "camera_id": cam_id,
                                    "boxes": boxes.tolist(),
                                    "scores": scores.tolist(),
                                    "labels": labels.tolist(),
                                }
                            )
                    detect_save_seconds += time.time() - save_started
            detection_time = time.time() - det_started
        gpu_total = time.time() - gpu_started

        result = {
            "gop_index": prepared.gop_index,
            "frame_ids": prepared.frame_ids,
            "active_frame_ids": prepared.active_frame_ids,
            "num_candidates": len(prepared.candidates),
            "num_placements": len(prepared.placements),
            "num_deferred": len(prepared.deferred),
            "bin_width": int(race_core.BIN_W),
            "bin_height": int(race_core.BIN_H),
            "sr_launch_plan": sr_launch_plan,
            "sr_launch_policy": self.sr_launch_policy,
            "sr_seconds": sr_time,
            "sr_materialize_seconds": sr_materialize_seconds,
            "sr_materialize_bin_init_seconds": sr_materialize_bin_init_seconds,
            "sr_materialize_patch_resize_seconds": sr_materialize_patch_resize_seconds,
            "sr_materialize_patch_paste_seconds": sr_materialize_patch_paste_seconds,
            "sr_copy_seconds": sr_copy_seconds,
            "sr_inference_seconds": sr_inference_seconds,
            "detect_seconds": detection_time,
            "detect_batch_plan": detect_batch_plan,
            "detect_engine_max_batch": self.detector_runner.max_batch,
            "detect_engine_opt_batch": self.detector_runner.opt_batch,
            "detect_input_count": len(detect_targets),
            "detect_frame_copy_seconds": detect_frame_copy_seconds,
            "detect_frame_cpu_stage_seconds": detect_frame_cpu_stage_seconds,
            "detect_frame_h2d_seconds": detect_frame_h2d_seconds,
            "detect_frame_gpu_convert_seconds": detect_frame_gpu_convert_seconds,
            "frame_transfer_mode": "direct",
            "detect_blend_seconds": detect_blend_seconds,
            # "detect_blend_to_tensor_seconds": detect_blend_to_tensor_seconds,
            # "detect_blend_upsample_seconds": detect_blend_upsample_seconds,
            # "detect_blend_patch_resize_seconds": detect_blend_patch_resize_seconds,
            # "detect_blend_patch_paste_seconds": detect_blend_patch_paste_seconds,
            "detect_to_tensor_seconds": detect_to_tensor_seconds,
            "detect_preprocess_seconds": detect_preprocess_seconds,
            "detect_trt_seconds": detect_trt_seconds,
            "detect_trt_setup_seconds": detect_trt_setup_seconds,
            "detect_trt_enqueue_seconds": detect_trt_enqueue_seconds,
            "detect_trt_sync_seconds": detect_trt_sync_seconds,
            "detect_parse_seconds": detect_parse_seconds,
            "detect_save_seconds": detect_save_seconds,
            "gpu_total_seconds": gpu_total,
            "worker_stats_by_camera": prepared.worker_stats,
            "candidate_snapshots": [_candidate_snapshot(candidate) for candidate in prepared.candidates],
            **prepared.cpu_stats,
            **prepared.area_stats,
        }
        self.runtime_stats.append(result)
        return result

    @staticmethod
    def format_gop_summary(summary: dict[str, Any]) -> str:
        worker_stats = summary.get("worker_stats_by_camera", {}) or {}
        if worker_stats:
            bottleneck_cam = max(
                worker_stats,
                key=lambda cam: worker_stats[cam].get("worker_total_seconds", 0.0),
            )
            bottleneck_stats = worker_stats[bottleneck_cam]
            worker_slow_ms = bottleneck_stats.get("worker_total_seconds", 0.0) * 1000.0
            worker_decode_ms = bottleneck_stats.get("decode_seconds", 0.0) * 1000.0
            worker_shm_write_ms = bottleneck_stats.get("shm_write_seconds", 0.0) * 1000.0
            worker_proposal_ms = bottleneck_stats.get("proposal_process_seconds", 0.0) * 1000.0
            worker_queue_put_ms = bottleneck_stats.get("queue_put_seconds", 0.0) * 1000.0
        else:
            bottleneck_cam = "single_process"
            worker_slow_ms = summary.get("decode_seconds", 0.0) * 1000.0 + summary.get("proposal_seconds", 0.0) * 1000.0
            worker_decode_ms = summary.get("decode_seconds", 0.0) * 1000.0
            worker_shm_write_ms = 0.0
            worker_proposal_ms = summary.get("proposal_seconds", 0.0) * 1000.0
            worker_queue_put_ms = 0.0

        proposal_collect_ms = summary.get("proposal_collect_seconds", 0.0) * 1000.0
        cpu_postprocess_ms = summary.get(
            "finalize_seconds",
            summary.get("match_seconds", 0.0)
            + summary.get("ranking_seconds", 0.0)
            + summary.get("packing_seconds", 0.0),
        ) * 1000.0
        frame_stage_ms = summary.get("detect_frame_cpu_stage_seconds", 0.0) * 1000.0
        frame_h2d_ms = summary.get("detect_frame_h2d_seconds", 0.0) * 1000.0
        frame_chw_ms = summary.get("detect_frame_gpu_convert_seconds", 0.0) * 1000.0
        frame_gpu_copy_ms = frame_h2d_ms + frame_chw_ms
        visible_cpu_ms = summary["cpu_total_seconds"] * 1000.0 + frame_stage_ms
        visible_gpu_ms = (
            frame_gpu_copy_ms
            + summary.get("sr_seconds", 0.0) * 1000.0
            + summary.get("detect_blend_seconds", 0.0) * 1000.0
            + summary.get("detect_seconds", 0.0) * 1000.0
        )
        total_ms = visible_cpu_ms + visible_gpu_ms
        active_frame_count = len(summary.get("active_frame_ids") or summary.get("frame_ids", []))
        fps = active_frame_count * 1000.0 / total_ms if total_ms > 0.0 else 0.0
        worker_overlapped_ms = max(0.0, worker_slow_ms - proposal_collect_ms)
        worker_other_ms = max(
            0.0,
            worker_slow_ms - worker_decode_ms - worker_shm_write_ms - worker_proposal_ms,
        )

        # Default log: compact critical-path view. The verbose legacy timing
        # block is kept below and can be re-enabled while profiling details.
        show_legacy_detail = False
        compact_lines = [
            (
                f"[GOP {summary['gop_index']:04d}] "
                "============================================"
            ),
            (
                f"frames={len(summary.get('frame_ids', []))} "
                f"props={summary.get('raw_proposal_count', 0)} "
                f"cand={summary.get('num_candidates', 0)} "
                f"placed={summary.get('num_placements', 0)} "
                f"bins={summary.get('opened_bin_count', 0)} "
                f"bin_size={summary.get('bin_width', 0)}x{summary.get('bin_height', 0)} "
                f"plan={summary.get('sr_launch_plan', [])} "
                f"sr_policy={summary.get('sr_launch_policy', 'profile')}"
            ),
            (
                f"packing_efficiency: sr_canvas_pixels={summary.get('sr_canvas_pixels', 0)} "
                f"placed_pixels={summary.get('placed_pixels', 0)} "
                f"fill_ratio={float(summary.get('fill_ratio', 0.0)):.2%}"
            ),
            (
                f"visible: cpu={visible_cpu_ms:.1f}ms "
                f"gpu={visible_gpu_ms:.1f}ms "
                f"total={total_ms:.1f}ms "
                f"fps={fps:.1f}"
            ),
            (
                f"cpu: proposal_collect={proposal_collect_ms:.1f}ms "
                f"cpu_postprocess={cpu_postprocess_ms:.1f}ms "
                f"(match={summary.get('match_seconds', 0.0) * 1000.0:.1f}ms "
                f"rank={summary.get('ranking_seconds', 0.0) * 1000.0:.1f}ms "
                f"pack={summary.get('packing_seconds', 0.0) * 1000.0:.1f}ms) "
                f"stage={frame_stage_ms:.1f}ms"
            ),
            (
                f"worker: overlapped={worker_overlapped_ms:.1f}ms "
                f"worker_slow={worker_slow_ms:.1f}ms "
                f"decode={worker_decode_ms:.1f}ms "
                f"shm_write={worker_shm_write_ms:.1f}ms "
                f"proposal={worker_proposal_ms:.1f}ms "
                f"other={worker_other_ms:.1f}ms "
                f"queue_put={worker_queue_put_ms:.1f}ms "
                f"bottleneck={bottleneck_cam}"
            ),
            (
                f"gpu: frame_copy={frame_gpu_copy_ms:.1f}ms "
                f"(h2d={frame_h2d_ms:.1f}ms "
                f"chw={frame_chw_ms:.1f}ms) "
                f"sr_total={_fmt_ms(summary.get('sr_seconds', 0.0))} "
                f"(bin_materialize={_fmt_ms(summary.get('sr_materialize_seconds', 0.0))} "
                f"pre_process={_fmt_ms(summary.get('sr_copy_seconds', 0.0))} "
                f"infer={_fmt_ms(summary.get('sr_inference_seconds', 0.0))})"
            ),
            (
                f"blend_frames={_fmt_ms(summary.get('detect_blend_seconds', 0.0))}"
            ),
            (
                f"gpu_detect: total={_fmt_ms(summary.get('detect_seconds', 0.0))} "
                f"plan={summary.get('detect_batch_plan', [])} "
                f"opt_batch={summary.get('detect_engine_opt_batch', 1)} "
                f"max_batch={summary.get('detect_engine_max_batch', 1)}"
            ),
            (
                f"to_tensor={_fmt_ms(summary.get('detect_to_tensor_seconds', 0.0))} "
                f"pre={_fmt_ms(summary.get('detect_preprocess_seconds', 0.0))} "
                f"trt={_fmt_ms(summary.get('detect_trt_seconds', 0.0))} "
                f"parse={_fmt_ms(summary.get('detect_parse_seconds', 0.0))} "
                f"save={_fmt_ms(summary.get('detect_save_seconds', 0.0))}"
            ),
            (
                f"gpu_detect_trt: setup={_fmt_ms(summary.get('detect_trt_setup_seconds', 0.0))} "
                f"enqueue={_fmt_ms(summary.get('detect_trt_enqueue_seconds', 0.0))} "
                f"sync={_fmt_ms(summary.get('detect_trt_sync_seconds', 0.0))}"
            ),
        ]
        if not show_legacy_detail:
            return "\n".join(compact_lines)

        sep = ">>>>>>>>>>>>>>"
        cpu_ms = summary["cpu_total_seconds"] * 1000.0
        gpu_ms = summary["gpu_total_seconds"] * 1000.0
        total_ms = max(cpu_ms, gpu_ms)
        worker_stats = summary.get("worker_stats_by_camera", {}) or {}
        if worker_stats:
            bottleneck_cam = max(
                worker_stats,
                key=lambda cam: worker_stats[cam].get("proposal_process_seconds", 0.0),
            )
            bottleneck_stats = worker_stats[bottleneck_cam]
            proposal_bgsub_ms = bottleneck_stats.get("bgsub_seconds", 0.0) * 1000.0
            proposal_gray_ms = bottleneck_stats.get("gray_seconds", 0.0) * 1000.0
            proposal_mog2_ms = bottleneck_stats.get("mog2_seconds", 0.0) * 1000.0
            proposal_morphology_ms = bottleneck_stats.get("morphology_seconds", 0.0) * 1000.0
            proposal_contour_ms = bottleneck_stats.get("contour_seconds", 0.0) * 1000.0
            proposal_consolidate_ms = bottleneck_stats.get("consolidate_seconds", 0.0) * 1000.0
            proposal_split_ms = bottleneck_stats.get("split_seconds", 0.0) * 1000.0
            proposal_materialize_ms = bottleneck_stats.get("materialize_seconds", 0.0) * 1000.0
            proposal_worker_ms = bottleneck_stats.get("proposal_process_seconds", 0.0) * 1000.0
            proposal_label = f"bottleneck_cam={bottleneck_cam}"
        else:
            proposal_bgsub_ms = summary["proposal_bgsub_seconds"] * 1000.0
            proposal_gray_ms = 0.0
            proposal_mog2_ms = proposal_bgsub_ms
            proposal_morphology_ms = summary["proposal_morphology_seconds"] * 1000.0
            proposal_contour_ms = summary["proposal_contour_seconds"] * 1000.0
            proposal_consolidate_ms = summary["proposal_consolidate_seconds"] * 1000.0
            proposal_split_ms = summary["proposal_split_seconds"] * 1000.0
            proposal_materialize_ms = summary["proposal_materialize_seconds"] * 1000.0
            proposal_worker_ms = summary["proposal_seconds"] * 1000.0
            proposal_label = "single_process"
        header = (
            f"[GOP {summary['gop_index']:04d}] "
            f"=========dedup={'on' if summary.get('cross_camera_dedup', True) else 'off'} "
            f"frames={len(summary['frame_ids'])}=============="
        )
        candidate_line = (
            f"raw_props={summary['raw_proposal_count']}, "
            f"cand={summary['num_candidates']}, "
            f"placed={summary['num_placements']}"
        )
        bin_line = (
            f"opened_bin_count={summary['opened_bin_count']} "
            f"max_sr_bins={summary.get('max_sr_bins')} "
            f"sr_launch_plan={summary['sr_launch_plan']}"
        )
        total_label = f"{sep}total_time={total_ms:.3f}ms"

        lines = [
            header,
            (
                f"decode_time={summary['decode_seconds'] * 1000.0:.3f}ms "
                f"proposal_time={summary['proposal_seconds'] * 1000.0:.3f}ms "
                f"match_time={summary['match_seconds'] * 1000.0:.3f}ms "
                f"ranking_time={summary['ranking_seconds'] * 1000.0:.3f}ms "
                f"packing_time={summary['packing_seconds'] * 1000.0:.3f}ms"
            ),
            f"{sep}proposal_time={summary['proposal_seconds'] * 1000.0:.3f}ms",
            (
                f"proposal_breakdown={proposal_label} "
                f"worker_proposal={proposal_worker_ms:.3f}ms "
                f"bgsub={proposal_bgsub_ms:.3f}ms "
                f"gray={proposal_gray_ms:.3f}ms "
                f"mog2={proposal_mog2_ms:.3f}ms "
                f"morphology={proposal_morphology_ms:.3f}ms "
                f"contour={proposal_contour_ms:.3f}ms"
            ),
            (
                f"proposal_consolidate={proposal_consolidate_ms:.3f}ms "
                f"proposal_split={proposal_split_ms:.3f}ms "
                f"proposal_materialize={proposal_materialize_ms:.3f}ms"
            ),
            f"proposal_sync_overhead={summary.get('proposal_sync_overhead_seconds', 0.0) * 1000.0:.3f}ms",
            (
                f"proposal_collect={summary.get('proposal_collect_seconds', 0.0) * 1000.0:.3f}ms "
                f"finalize={summary.get('finalize_seconds', 0.0) * 1000.0:.3f}ms "
                f"cpu_stage={summary.get('cpu_stage_seconds', summary['cpu_total_seconds']) * 1000.0:.3f}ms "
                f"worker_hidden={summary.get('worker_hidden_seconds', 0.0) * 1000.0:.3f}ms"
            ),
        ]
        if worker_stats:
            collect_total_seconds = max(
                (worker_stats[cam].get("collect_total_seconds", 0.0) for cam in worker_stats),
                default=0.0,
            )
            lines.extend(
                [
                    "worker_total=" + " ".join(
                        f"{cam}:{worker_stats[cam].get('worker_total_seconds', 0.0) * 1000.0:.3f}ms"
                        for cam in sorted(worker_stats)
                    ),
                    "worker_decode=" + " ".join(
                        f"{cam}:{worker_stats[cam].get('decode_seconds', 0.0) * 1000.0:.3f}ms"
                        for cam in sorted(worker_stats)
                    ),
                    "worker_proposal=" + " ".join(
                        f"{cam}:{worker_stats[cam].get('proposal_process_seconds', 0.0) * 1000.0:.3f}ms"
                        for cam in sorted(worker_stats)
                    ),
                    "worker_queue_put=" + " ".join(
                        f"{cam}:{worker_stats[cam].get('queue_put_seconds', 0.0) * 1000.0:.3f}ms"
                        for cam in sorted(worker_stats)
                    ),
                    "main_queue_get=" + " ".join(
                        f"{cam}:{worker_stats[cam].get('queue_get_seconds', 0.0) * 1000.0:.3f}ms"
                        for cam in sorted(worker_stats)
                    ),
                    "shared_frame_mb=" + " ".join(
                        f"{cam}:{worker_stats[cam].get('frame_bytes', 0.0) / (1024.0 * 1024.0):.2f}MB"
                        for cam in sorted(worker_stats)
                    )
                    + f" collect_total={collect_total_seconds * 1000.0:.3f}ms",
                ]
            )
        lines.extend(
            [
                f"{sep}packing_time={summary['packing_seconds'] * 1000.0:.3f}ms",
                (
                    f"candidate_sort_time={summary.get('candidate_sort_seconds', 0.0) * 1000.0:.3f}ms "
                    f"admission_time={summary.get('admission_seconds', 0.0) * 1000.0:.3f}ms "
                    f"compact_pack_time={summary.get('compact_pack_seconds', 0.0) * 1000.0:.3f}ms "
                    f"render_bin_time={summary.get('render_bin_seconds', 0.0) * 1000.0:.3f}ms"
                ),
                (
                    f"repack_count={summary.get('repack_count', 0)} "
                    f"free_rect_count={summary.get('free_rect_count', 0)}"
                ),
                (
                    f"{sep}sr_time={summary['sr_seconds'] * 1000.0:.3f}ms "
                    f"sr_bin_materialize={summary.get('sr_materialize_seconds', 0.0) * 1000.0:.3f}ms "
                    f"sr_pre_process={summary.get('sr_copy_seconds', 0.0) * 1000.0:.3f}ms "
                    f"sr_inference={summary.get('sr_inference_seconds', 0.0) * 1000.0:.3f}ms"
                ),
                (
                    f"sr_materialize_bin_init={summary.get('sr_materialize_bin_init_seconds', 0.0) * 1000.0:.3f}ms "
                    f"sr_materialize_patch_resize={summary.get('sr_materialize_patch_resize_seconds', 0.0) * 1000.0:.3f}ms "
                    f"sr_materialize_patch_paste={summary.get('sr_materialize_patch_paste_seconds', 0.0) * 1000.0:.3f}ms"
                ),
                bin_line,
                f"{sep}detect_time={summary['detect_seconds'] * 1000.0:.3f}ms",
                (
                    f"detect_inputs={summary.get('detect_input_count', 0)} "
                    f"detect_batch_plan={summary.get('detect_batch_plan', [])} "
                    f"detect_max_batch={summary.get('detect_engine_max_batch', 1)}"
                ),
                (
                    f"detect_frame_copy={summary.get('detect_frame_copy_seconds', 0.0) * 1000.0:.3f}ms "
                    f"detect_frame_cpu={summary.get('detect_frame_cpu_stage_seconds', 0.0) * 1000.0:.3f}ms "
                    f"detect_frame_h2d={summary.get('detect_frame_h2d_seconds', 0.0) * 1000.0:.3f}ms "
                    f"detect_frame_convert={summary.get('detect_frame_gpu_convert_seconds', 0.0) * 1000.0:.3f}ms "
                    f"blend_frames={summary.get('detect_blend_seconds', 0.0) * 1000.0:.3f}ms "
                    f"detect_to_tensor={summary.get('detect_to_tensor_seconds', 0.0) * 1000.0:.3f}ms "
                    f"detect_preprocess={summary.get('detect_preprocess_seconds', 0.0) * 1000.0:.3f}ms "
                    f"detect_trt={summary.get('detect_trt_seconds', 0.0) * 1000.0:.3f}ms "
                    f"detect_trt_setup={summary.get('detect_trt_setup_seconds', 0.0) * 1000.0:.3f}ms "
                    f"detect_trt_enqueue={summary.get('detect_trt_enqueue_seconds', 0.0) * 1000.0:.3f}ms "
                    f"detect_trt_sync={summary.get('detect_trt_sync_seconds', 0.0) * 1000.0:.3f}ms "
                    f"detect_parse={summary.get('detect_parse_seconds', 0.0) * 1000.0:.3f}ms "
                    f"detect_save={summary.get('detect_save_seconds', 0.0) * 1000.0:.3f}ms"
                ),
                total_label,
                candidate_line,
                f"cpu={cpu_ms:.3f}ms gpu={gpu_ms:.3f}ms",
            ]
        )
        # Uncomment when blend_back_frames_torch(..., return_timing=True) is enabled.
        # line4c = (
        #     f"blend_to_tensor={summary.get('detect_blend_to_tensor_seconds', 0.0) * 1000.0:.3f}ms "
        #     f"blend_upsample={summary.get('detect_blend_upsample_seconds', 0.0) * 1000.0:.3f}ms "
        #     f"blend_patch_resize={summary.get('detect_blend_patch_resize_seconds', 0.0) * 1000.0:.3f}ms "
        #     f"blend_patch_paste={summary.get('detect_blend_patch_paste_seconds', 0.0) * 1000.0:.3f}ms"
        # )
        return "\n".join(lines)

    def run(self, *, start_frame: int = 1, num_frames: int | None = None) -> list[dict[str, Any]]:
        """Run GOPs until `num_frames` is exhausted.

        In worker mode, proposal generation for GOP k+1 overlaps with CPU/GPU
        processing of GOP k. Finalize and GPU work are still serialized in the
        main process, which is why visible GOP time is reported as CPU + GPU.
        """

        self.seek(start_frame)
        limit = None if num_frames is None else start_frame + num_frames - 1
        summaries: list[dict[str, Any]] = []

        if self._proposal_pool is not None:
            next_start = start_frame
            current_job = self._submit_worker_gop(
                next_start,
                limit,
                self._gop_counter,
            )
            while current_job is not None:
                raw = self._collect_worker_gop(current_job)
                if raw is None:
                    break

                next_start = raw.frame_ids[-1] + 1
                if limit is not None and next_start > limit:
                    next_job = None
                else:
                    next_job = self._submit_worker_gop(
                        next_start,
                        limit,
                        raw.gop_index + 1,
                    )

                prepared = self._finalize_raw_gop(raw, current_job.submitted_at)
                self._gop_counter = raw.gop_index + 1
                summary = self.process_gop(prepared)
                summaries.append(summary)
                print(self.format_gop_summary(summary), flush=True)
                current_job = next_job

            self._write_runtime_outputs()
            return summaries

        next_start = start_frame

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.prepare_gop, next_start, limit)
            while future is not None:
                prepared = future.result()
                if prepared is None:
                    break

                self._gop_counter += 1
                next_start = prepared.frame_ids[-1] + 1
                if limit is not None and next_start > limit:
                    next_future = None
                else:
                    next_future = executor.submit(self.prepare_gop, next_start, limit)

                summary = self.process_gop(prepared)
                summaries.append(summary)
                print(self.format_gop_summary(summary), flush=True)
                future = next_future

        self._write_runtime_outputs()
        return summaries

    def _write_runtime_outputs(self) -> None:
        stats_path = os.path.join(self.output_dir, "runtime_stats.json")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(stats_path).write_text(json.dumps(self.runtime_stats, indent=2), encoding="utf-8")
        detections_path = os.path.join(self.output_dir, "detections.json")
        Path(detections_path).write_text(json.dumps(self.detection_records, indent=2), encoding="utf-8")
        if self.proposal_cache_path and self.proposal_generator is not None:
            save_proposal_artifact(
                self.proposal_cache_path,
                self.proposal_generator.generated,
                cameras=self.cameras,
                frame_size=(640, 360),
                metadata={
                    "mode": "online_generated_runtime_cache",
                    "warmup_frames": self.proposal_generator.cfg["warmup_frames"],
                },
            )

def homography_loader_cameras(loader: HomographyArtifactLoader) -> list[str]:
    if loader.cameras:
        return list(loader.cameras)
    cameras = set()
    for pair in loader.iter_pairs():
        cameras.add(pair.src_cam)
        cameras.add(pair.ref_cam)
    return sorted(cameras)
