from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

IMG_W, IMG_H = 640, 360
IMG_SIZE = IMG_W * IMG_H
BIN_W, BIN_H = 256, 256
SCALE = 3
K_TOTAL = 4
EPS = 1e-6
WS, WC = 0.48, 0.52
MU = 0.5
ADMISSION_FILL_RATIO = 0.95


def set_bin_size(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError(f"bin size must be positive, got {width}x{height}")
    global BIN_W, BIN_H
    BIN_W, BIN_H = int(width), int(height)


@dataclass
class CandidatePatch:
    """Enhancement candidate selected from a matched object cluster.

    `patch` is intentionally optional. In the optimized runtime the CPU keeps
    only bbox metadata, then the GPU crops the patch from the already-copied
    full frame during bin materialization.
    """

    cluster_id: str
    cam_id: str
    frame_id: int
    patch: np.ndarray | None
    orig_bbox: tuple[int, int, int, int]
    importance: float
    metadata: dict[str, Any] = field(default_factory=dict)
    patch_w: int | None = None
    patch_h: int | None = None

    @property
    def w(self) -> int:
        if self.patch is not None:
            return int(self.patch.shape[1])
        if self.patch_w is not None:
            return int(self.patch_w)
        return max(0, int(self.orig_bbox[2]))

    @property
    def h(self) -> int:
        if self.patch is not None:
            return int(self.patch.shape[0])
        if self.patch_h is not None:
            return int(self.patch_h)
        return max(0, int(self.orig_bbox[3]))


@dataclass
class PlacementEntry:
    """Mapping from one candidate bbox to its location inside an SR bin."""

    cluster_id: str
    cam_id: str
    frame_id: int
    bin_idx: int
    bx: int
    by: int
    bw: int
    bh: int
    orig_bbox: tuple[int, int, int, int]
    orig_w: int
    orig_h: int
    importance: float
    metadata: dict[str, Any] = field(default_factory=dict)


def crop_patch(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    return frame[y1:y2, x1:x2].copy()


def compute_blur(patch: np.ndarray) -> float:
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_view_quality(
    frames: dict[str, np.ndarray],
    obj_views: dict[str, tuple[int, int, int, int]],
    debug_label: str | None = None,
) -> dict[str, float]:
    """Score each camera view by object size and local sharpness."""

    s_raw, c_raw = {}, {}

    for cam, (x, y, w, h) in obj_views.items():
        if cam not in frames:
            continue
        obj_size = w * h
        s_raw[cam] = np.log(obj_size / IMG_SIZE + EPS)

        patch = crop_patch(frames[cam], (x, y, w, h))
        c_raw[cam] = compute_blur(patch) if patch.size else 0.0


    if not s_raw:
        return {}

    # def normalize(values: dict[str, float]) -> dict[str, float]:
    #     arr = np.asarray(list(values.values()), dtype=np.float32)
    #     mn, mx = float(arr.min()), float(arr.max())
    #     return {k: (float(v) - mn) / (mx - mn + EPS) for k, v in values.items()}
    def normalize(values: dict[str, float]) -> dict[str, float]:
        arr = np.asarray(list(values.values()), dtype=np.float32)
        mn, mx = float(arr.min()), float(arr.max())
        denom = mx - mn
        if denom <= EPS:
            return {k: 0.5 for k in values}
        return {
            k: float(np.clip((float(v) - mn) / denom, 0.0, 1.0))
            for k, v in values.items()
        }

    s_norm = normalize(s_raw)
    c_norm = normalize(c_raw)
    q_scores = {
        cam: float(np.clip(WS * s_norm[cam] + WC * c_norm[cam], 0.0, 1.0))
        for cam in s_norm
    }
    # if debug_label:
    #     print(
    #         "[view_quality] "
    #         f"{debug_label} "
    #         f"s_raw={s_raw} "
    #         f"c_raw={c_raw} "
    #         f"s_norm={s_norm} "
    #         f"c_norm={c_norm} "
    #         f"q_scores={q_scores}",
    #         flush=True,
    #     )
    return q_scores


def select_best_view(Q_scores: dict[str, float]) -> str:
    return min(Q_scores, key=Q_scores.get)


def compute_importance(
    obj_views: dict[str, tuple[int, int, int, int]],
    Q_scores: dict[str, float],
    best_cam: str,
    debug_label: str | None = None,
) -> float:
    """Estimate SR utility per bin pixel for a cross-camera object cluster."""

    k = len(obj_views)
    redundancy_gain = (k - 1) / k

    q_best = float(np.clip(Q_scores.get(best_cam, 1.0), 0.0, 1.0))
    # q_best = float(Q_scores.get(best_cam, 1.0))
    enhancement_gain = max(0.0, 1.0 - q_best)

    _, _, w, h = obj_views[best_cam]
    obj_area = max(1, int(w)) * max(1, int(h))
    sr_cost = max(obj_area / float(BIN_W * BIN_H), EPS)
    utility = MU * redundancy_gain + (1.0 - MU) * enhancement_gain
    importance = max(0.0, utility) / sr_cost
    # if debug_label:
    #     print(
    #         "[importance_raw] "
    #         f"{debug_label} "
    #         f"k={k} "
    #         f"redundancy_gain={redundancy_gain:.6f} "
    #         f"q_best={q_best:.6f} "
    #         f"enhancement_gain={enhancement_gain:.6f} "
    #         f"obj_area={obj_area} "
    #         f"sr_cost={sr_cost:.6f} "
    #         f"utility={utility:.6f} "
    #         f"importance_raw={importance:.6f}",
    #         flush=True,
    #     )
    return importance


def normalize_candidate_importance(candidates: list[CandidatePatch]) -> None:
    """Normalize candidate priorities into [0, 1] before admission/packing."""

    if not candidates:
        return
    scores = np.asarray(
        [
            float(candidate.importance)
            if np.isfinite(float(candidate.importance)) and float(candidate.importance) > 0.0
            else 0.0
            for candidate in candidates
        ],
        dtype=np.float32,
    )
    mn, mx = float(scores.min()), float(scores.max())
    if mx - mn <= EPS:
        normalized = np.ones_like(scores, dtype=np.float32) if mx > 0.0 else np.zeros_like(scores, dtype=np.float32)
    else:
        normalized = (scores - mn) / (mx - mn)
    for candidate, score in zip(candidates, normalized):
        candidate.importance = float(np.clip(score, 0.0, 1.0))


def _estimated_bin_area(candidate: CandidatePatch) -> int:
    ow, oh = candidate.w, candidate.h
    if ow <= 0 or oh <= 0:
        return 0
    if ow > BIN_W or oh > BIN_H:
        scale = min(BIN_W / float(ow), BIN_H / float(oh))
        ow = max(1, int(round(ow * scale)))
        oh = max(1, int(round(oh * scale)))
    return int(ow) * int(oh)


def select_candidates_under_bin_budget(
    candidates: Iterable[CandidatePatch],
    max_bins: int | None,
    profile: dict[str, float | int] | None = None,
) -> tuple[list[CandidatePatch], list[CandidatePatch]]:
    """Keep the most valuable candidates that can fit under the SR bin budget."""

    candidates = list(candidates)
    if max_bins is None or max_bins <= 0:
        return candidates, []

    selected: list[CandidatePatch] = []
    deferred: list[CandidatePatch] = []
    t_sort = time.perf_counter()
    queue = sorted(
        candidates,
        key=lambda c: (
            -float(c.importance),
            -(c.w * c.h),
            c.frame_id,
            c.cam_id,
            c.cluster_id,
        ),
    )
    if profile is not None:
        profile["candidate_sort_seconds"] = float(profile.get("candidate_sort_seconds", 0.0)) + (time.perf_counter() - t_sort)

    t_admission = time.perf_counter()
    budget_area = float(max_bins * BIN_W * BIN_H) * ADMISSION_FILL_RATIO
    used_area = 0.0
    for candidate in queue:
        area = float(_estimated_bin_area(candidate))
        if selected and used_area + area > budget_area:
            deferred.append(candidate)
            continue
        selected.append(candidate)
        used_area += area

    bins, _, invalid = pack_objects(selected, profile=profile)
    if invalid:
        invalid_ids = {id(candidate) for candidate in invalid}
        selected = [candidate for candidate in selected if id(candidate) not in invalid_ids]
        deferred.extend(invalid)
        bins, _, _ = pack_objects(selected, profile=profile) if selected else ([], [], [])

    while len(bins) > max_bins and selected:
        victim_idx = min(
            range(len(selected)),
            key=lambda idx: (
                float(selected[idx].importance),
                _estimated_bin_area(selected[idx]),
                -selected[idx].frame_id,
                selected[idx].cam_id,
                selected[idx].cluster_id,
            ),
        )
        victim = selected.pop(victim_idx)
        deferred.append(victim)
        bins, _, invalid = pack_objects(selected, profile=profile) if selected else ([], [], [])
        if invalid:
            invalid_ids = {id(candidate) for candidate in invalid}
            selected = [candidate for candidate in selected if id(candidate) not in invalid_ids]
            deferred.extend(invalid)
            bins, _, _ = pack_objects(selected, profile=profile) if selected else ([], [], [])

    if profile is not None:
        profile["admission_seconds"] = float(profile.get("admission_seconds", 0.0)) + (time.perf_counter() - t_admission)

    return selected, deferred


class Bin:
    """Max-rects style 2D bin used to compact candidate patches for SR."""

    def __init__(self, bin_w: int, bin_h: int):
        self.w = bin_w
        self.h = bin_h
        self.image = np.zeros((bin_h, bin_w, 3), dtype=np.uint8)
        self.free = [(0, 0, bin_w, bin_h)]

    def placement_cost(self, ow: int, oh: int, rx: int, ry: int, rw: int, rh: int) -> tuple[float, float, float, int, int]:
        c_area = (rw * rh - ow * oh) / float(self.w * self.h)
        c_short = min(rw - ow, rh - oh) / float(max(self.w, self.h))
        c_long = max(rw - ow, rh - oh) / float(max(self.w, self.h))
        return (c_area, c_short, c_long, ry, rx)

    @staticmethod
    def fits(ow: int, oh: int, rw: int, rh: int) -> bool:
        return ow <= rw and oh <= rh

    def place(
        self,
        patch: np.ndarray | None,
        ox: int,
        oy: int,
        ow: int,
        oh: int,
        *,
        profile: dict[str, float | int] | None = None,
    ) -> None:
        if patch is not None:
            t_render = time.perf_counter()
            self.image[oy:oy + oh, ox:ox + ow] = patch[:oh, :ow]
            if profile is not None:
                profile["render_bin_seconds"] = float(profile.get("render_bin_seconds", 0.0)) + (time.perf_counter() - t_render)
        new_free = []
        placed = (ox, oy, ow, oh)
        for rx, ry, rw, rh in self.free:
            if not _rects_overlap((rx, ry, rw, rh), placed):
                new_free.append((rx, ry, rw, rh))
                continue
            new_free.extend(_split_free_rect((rx, ry, rw, rh), placed))
        self.free = _prune_free_rects(new_free)

def _prepare_patch_for_bin(candidate: CandidatePatch) -> tuple[np.ndarray | None, int, int] | None:
    patch = candidate.patch
    ow, oh = candidate.w, candidate.h

    if ow <= 0 or oh <= 0:
        return None

    if ow > BIN_W or oh > BIN_H:
        scale = min(BIN_W / float(ow), BIN_H / float(oh))
        ow = max(1, int(round(ow * scale)))
        oh = max(1, int(round(oh * scale)))
        if patch is not None:
            patch = cv2.resize(patch, (ow, oh), interpolation=cv2.INTER_LINEAR)

    return patch, ow, oh


def _prune_free_rects(rects: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    deduped = list(dict.fromkeys(rect for rect in rects if rect[2] > 0 and rect[3] > 0))
    cleaned = deduped
    pruned: list[tuple[int, int, int, int]] = []
    for idx, rect in enumerate(cleaned):
        x, y, w, h = rect
        contained = False
        for jdx, other in enumerate(cleaned):
            if idx == jdx:
                continue
            ox, oy, ow, oh = other
            if x >= ox and y >= oy and x + w <= ox + ow and y + h <= oy + oh:
                contained = True
                break
        if not contained:
            pruned.append(rect)
    return sorted(pruned, key=lambda item: (item[1], item[0], item[2] * item[3], item[2], item[3]))


def _rects_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


def _split_free_rect(
    free_rect: tuple[int, int, int, int],
    used_rect: tuple[int, int, int, int],
) -> list[tuple[int, int, int, int]]:
    fx, fy, fw, fh = free_rect
    ux, uy, uw, uh = used_rect
    fx2, fy2 = fx + fw, fy + fh
    ux2, uy2 = ux + uw, uy + uh
    splits: list[tuple[int, int, int, int]] = []

    if ux > fx:
        splits.append((fx, fy, ux - fx, fh))
    if ux2 < fx2:
        splits.append((ux2, fy, fx2 - ux2, fh))
    if uy > fy:
        splits.append((fx, fy, fw, uy - fy))
    if uy2 < fy2:
        splits.append((fx, uy2, fw, fy2 - uy2))

    return splits


def _find_best_placement(
    bins: list[Bin],
    *,
    ow: int,
    oh: int,
) -> tuple[int, int, int, int, int] | None:
    best: tuple[int, int, int, int, int] | None = None
    best_cost: tuple[float, float, float, int, int] | None = None

    for bi, current_bin in enumerate(bins):
        for rx, ry, rw, rh in current_bin.free:
            if not current_bin.fits(ow, oh, rw, rh):
                continue
            cost = current_bin.placement_cost(ow, oh, rx, ry, rw, rh)
            if best_cost is None or cost < best_cost:
                best = (bi, rx, ry, ow, oh)
                best_cost = cost

    return best


def pack_objects(
    candidates: Iterable[CandidatePatch],
    num_bins: int | None = None,
    profile: dict[str, float | int] | None = None,
) -> tuple[list[Bin], list[PlacementEntry], list[CandidatePatch]]:
    """Pack candidates into lazily-created SR bins and return placement maps."""

    _ = num_bins
    pack_started = time.perf_counter()
    render_before = float(profile.get("render_bin_seconds", 0.0)) if profile is not None else 0.0
    if profile is not None:
        profile["repack_count"] = int(profile.get("repack_count", 0)) + 1

    bins: list[Bin] = []
    placements: list[PlacementEntry] = []
    deferred: list[CandidatePatch] = []

    queue = sorted(
        candidates,
        key=lambda c: (
            -(c.w * c.h),
            -max(c.w, c.h),
            -float(c.importance),
            c.frame_id,
            c.cam_id,
            c.cluster_id,
        ),
    )

    for candidate in queue:
        prepared = _prepare_patch_for_bin(candidate)
        if prepared is None:
            deferred.append(candidate)
            continue
        patch, ow, oh = prepared

        best = _find_best_placement(bins, ow=ow, oh=oh)
        if best is None:
            bins.append(Bin(BIN_W, BIN_H))
            best = _find_best_placement([bins[-1]], ow=ow, oh=oh)
            if best is None:
                bins.pop()
                deferred.append(candidate)
                continue
            _, bx, by, bw, bh = best
            bi = len(bins) - 1
        else:
            bi, bx, by, bw, bh = best

        bins[bi].place(patch, bx, by, bw, bh, profile=profile)
        placements.append(
            PlacementEntry(
                cluster_id=candidate.cluster_id,
                cam_id=candidate.cam_id,
                frame_id=candidate.frame_id,
                bin_idx=bi,
                bx=bx,
                by=by,
                bw=bw,
                bh=bh,
                orig_bbox=candidate.orig_bbox,
                orig_w=candidate.w,
                orig_h=candidate.h,
                importance=float(candidate.importance),
                metadata=dict(candidate.metadata),
            )
        )

    if profile is not None:
        render_after = float(profile.get("render_bin_seconds", 0.0))
        compact_elapsed = max(0.0, time.perf_counter() - pack_started - (render_after - render_before))
        profile["compact_pack_seconds"] = float(profile.get("compact_pack_seconds", 0.0)) + compact_elapsed
        profile["free_rect_count"] = sum(len(current_bin.free) for current_bin in bins)

    return bins, placements, deferred


def optimize_launch_plan(
    num_bins: int,
    batch_latency_ms: dict[int, float] | None,
) -> list[int]:
    """Choose an SR batch split that minimizes profiled latency for K bins."""

    if num_bins <= 0:
        return []
    if not batch_latency_ms:
        return [num_bins]

    supported = sorted(int(batch_size) for batch_size in batch_latency_ms if int(batch_size) > 0)
    if not supported:
        return [num_bins]

    inf = float("inf")
    best_cost = [inf] * (num_bins + 1)
    best_steps = [10**9] * (num_bins + 1)
    best_plan: list[list[int] | None] = [None] * (num_bins + 1)
    best_cost[0] = 0.0
    best_steps[0] = 0
    best_plan[0] = []

    for total in range(1, num_bins + 1):
        for batch_size in sorted(supported, reverse=True):
            if batch_size > total:
                continue
            prev_plan = best_plan[total - batch_size]
            if prev_plan is None:
                continue
            candidate_cost = best_cost[total - batch_size] + float(batch_latency_ms[batch_size])
            candidate_steps = best_steps[total - batch_size] + 1
            candidate_plan = prev_plan + [batch_size]
            current_plan = best_plan[total]
            if (
                candidate_cost < best_cost[total] - 1e-9
                or (
                    abs(candidate_cost - best_cost[total]) <= 1e-9
                    and (
                        candidate_steps < best_steps[total]
                        or (
                            candidate_steps == best_steps[total]
                            and (current_plan is None or tuple(candidate_plan) > tuple(current_plan))
                        )
                    )
                )
            ):
                best_cost[total] = candidate_cost
                best_steps[total] = candidate_steps
                best_plan[total] = candidate_plan

    if best_plan[num_bins] is None:
        raise ValueError(
            f"Could not cover {num_bins} bins with supported batch sizes {sorted(supported)}"
        )
    return best_plan[num_bins] or []


def bilinear_upscale(frame: np.ndarray, scale: int = SCALE) -> np.ndarray:
    return cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_LINEAR)


def _as_bgr_tensor(
    frame: np.ndarray | torch.Tensor,
    *,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    if torch.is_tensor(frame):
        tensor = frame
        if tensor.ndim == 3 and tensor.shape[0] != 3 and tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)
        if tensor.ndim != 3 or tensor.shape[0] != 3:
            raise ValueError(f"Expected CHW or HWC 3-channel tensor, got shape={tuple(tensor.shape)}")
        if device is not None:
            tensor = tensor.to(device=device)
        return tensor.float().contiguous()

    tensor = torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1)))
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor.float().contiguous()


def _as_bgr_batch_tensor(
    frames: list[np.ndarray | torch.Tensor] | torch.Tensor,
    *,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    if torch.is_tensor(frames):
        tensor = frames
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim == 4 and tensor.shape[1] != 3 and tensor.shape[-1] == 3:
            tensor = tensor.permute(0, 3, 1, 2)
        if tensor.ndim != 4 or tensor.shape[1] != 3:
            raise ValueError(f"Expected BCHW or BHWC 3-channel tensor batch, got shape={tuple(tensor.shape)}")
        if device is not None:
            tensor = tensor.to(device=device)
        return tensor.float().contiguous()

    if not frames:
        raise ValueError("Expected at least one frame")
    if all(isinstance(frame, np.ndarray) for frame in frames):
        first = frames[0]
        if first.ndim != 3 or first.shape[-1] != 3:
            raise ValueError(f"Expected HWC 3-channel frames, got shape={first.shape}")
        for frame in frames:
            if frame.shape != first.shape:
                raise ValueError(f"Expected same frame shape in batch, got {frame.shape} and {first.shape}")
        batch = np.stack([np.ascontiguousarray(frame) for frame in frames], axis=0)
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2)
        if device is not None:
            tensor = tensor.to(device=device)
        return tensor.float().contiguous()
    tensors = [_as_bgr_tensor(frame, device=device) for frame in frames]
    return torch.stack(tensors, dim=0).float().contiguous()


def tensor_to_bgr_image(frame: torch.Tensor) -> np.ndarray:
    tensor = frame.detach()
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"Expected CHW tensor, got shape={tuple(tensor.shape)}")
    image = tensor.clamp(0, 255).byte().cpu().permute(1, 2, 0).numpy()
    return np.ascontiguousarray(image)


def run_sr_batch(
    sr_model: Any,
    bin_images: list[np.ndarray | torch.Tensor],
    *,
    device: str | torch.device | None = None,
    launch_plan: list[int] | None = None,
    return_tensors: bool = False,
    return_timing: bool = False,
) -> list[np.ndarray] | tuple[list[np.ndarray], dict[str, float]]:
    """Convert packed BGR bins to RGB tensor chunks, run SR, then restore BGR."""

    if not bin_images:
        empty_timing = {"copy_seconds": 0.0, "inference_seconds": 0.0}
        return ([], empty_timing) if return_timing else []

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    copy_started = time.time()
    if all(torch.is_tensor(img) for img in bin_images):
        batch = torch.stack(
            [
                (
                    img[[2, 1, 0], :, :]
                    if img.ndim == 3 and img.shape[0] == 3
                    else img.permute(2, 0, 1)[[2, 1, 0], :, :]
                ).float()
                for img in bin_images
            ],
            dim=0,
        )
        batch = batch.to(device=device, dtype=torch.float32)
    else:
        rgb = np.stack([cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB).astype(np.float32) for img in bin_images], axis=0)
        batch = torch.from_numpy(rgb).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    copy_seconds = time.time() - copy_started
    if launch_plan is None:
        launch_plan = [len(bin_images)]
    if sum(int(item) for item in launch_plan) != len(bin_images):
        raise ValueError(
            f"Launch plan {launch_plan} does not cover {len(bin_images)} packed bins"
        )

    results: list[Any] = []
    offset = 0
    inference_seconds = 0.0
    for batch_size in launch_plan:
        chunk = batch[offset:offset + batch_size].contiguous()
        infer_started = time.time()
        chunk_results = sr_model.inference(chunk)
        inference_seconds += time.time() - infer_started
        if torch.is_tensor(chunk_results):
            results.extend(chunk_results)
        else:
            results.extend(list(chunk_results))
        offset += batch_size

    sr_bins = []
    for item in results:
        tensor = item if torch.is_tensor(item) else torch.as_tensor(item)
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        bgr_tensor = tensor[[2, 1, 0], :, :].contiguous()
        if return_tensors:
            sr_bins.append(bgr_tensor)
        else:
            sr_bins.append(tensor_to_bgr_image(bgr_tensor))
    if return_timing:
        return sr_bins, {
            "copy_seconds": copy_seconds,
            "inference_seconds": inference_seconds,
        }
    return sr_bins


def blend_back_frame(
    frame_lr: np.ndarray,
    sr_bins: list[np.ndarray],
    placements: Iterable[PlacementEntry],
    *,
    scale: int = SCALE,
) -> np.ndarray: 
    """CPU reference path: upscale the frame and paste enhanced SR patches."""

    frame_hr = bilinear_upscale(frame_lr, scale=scale)
    for entry in placements:
        if entry.bin_idx >= len(sr_bins):
            continue
        sr_bin = sr_bins[entry.bin_idx]
        sx1, sy1 = entry.bx * scale, entry.by * scale
        sx2, sy2 = sx1 + entry.bw * scale, sy1 + entry.bh * scale
        sr_patch = sr_bin[sy1:sy2, sx1:sx2]
        x, y, w, h = entry.orig_bbox
        dx1, dy1 = x * scale, y * scale
        dx2 = min(frame_hr.shape[1], (x + w) * scale)
        dy2 = min(frame_hr.shape[0], (y + h) * scale)
        dw, dh = dx2 - dx1, dy2 - dy1
        if dw <= 0 or dh <= 0 or sr_patch.size == 0:
            continue
        frame_hr[dy1:dy2, dx1:dx2] = cv2.resize(sr_patch, (dw, dh), interpolation=cv2.INTER_LINEAR)
    return frame_hr


def blend_back_frame_torch(
    frame_lr: np.ndarray | torch.Tensor,
    sr_bins: list[torch.Tensor],
    placements: Iterable[PlacementEntry],
    *,
    scale: int = SCALE,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """GPU single-frame blend path used by tests and debugging utilities."""

    frame_tensor = _as_bgr_tensor(frame_lr, device=device)
    frame_hr = F.interpolate(
        frame_tensor.unsqueeze(0),
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    for entry in placements:
        if entry.bin_idx >= len(sr_bins):
            continue
        sr_bin = sr_bins[entry.bin_idx]
        sx1, sy1 = entry.bx * scale, entry.by * scale
        sx2, sy2 = sx1 + entry.bw * scale, sy1 + entry.bh * scale
        sr_patch = sr_bin[:, sy1:sy2, sx1:sx2]
        x, y, w, h = entry.orig_bbox
        dx1, dy1 = x * scale, y * scale 
        dx2 = min(frame_hr.shape[2], (x + w) * scale)
        dy2 = min(frame_hr.shape[1], (y + h) * scale)
        dw, dh = dx2 - dx1, dy2 - dy1
        if dw <= 0 or dh <= 0 or sr_patch.numel() == 0:
            continue
        if sr_patch.shape[1] != dh or sr_patch.shape[2] != dw:
            sr_patch = F.interpolate(
                sr_patch.unsqueeze(0),
                size=(dh, dw),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        frame_hr[:, dy1:dy2, dx1:dx2] = sr_patch
    return frame_hr


def blend_back_frames_torch(
    frames_lr: list[np.ndarray | torch.Tensor] | torch.Tensor,
    sr_bins: list[torch.Tensor],
    placements_by_frame: Iterable[Iterable[PlacementEntry]],
    *,
    scale: int = SCALE,
    device: str | torch.device | None = None,
    return_timing: bool = False,
) -> list[torch.Tensor] | tuple[list[torch.Tensor], dict[str, float]]:
    """Batch-upsample LR frames and paste all SR patches before detection."""

    placements_list = [list(placements) for placements in placements_by_frame]
    frame_count = int(frames_lr.shape[0]) if torch.is_tensor(frames_lr) and frames_lr.ndim >= 4 else len(frames_lr)
    if frame_count != len(placements_list):
        raise ValueError(
            f"Expected one placement list per frame, got {frame_count} frames "
            f"and {len(placements_list)} placement lists"
        )

    timing = {
        "to_tensor_seconds": 0.0,
        "upsample_seconds": 0.0,
        "patch_resize_seconds": 0.0,
        "patch_paste_seconds": 0.0,
    }
    if frame_count == 0:
        return ([], timing) if return_timing else []

    started = time.time()
    frame_batch = _as_bgr_batch_tensor(frames_lr, device=device)
    timing["to_tensor_seconds"] = time.time() - started

    started = time.time()
    frame_hr_batch = F.interpolate(
        frame_batch,
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
    )
    timing["upsample_seconds"] = time.time() - started

    for frame_idx, placements in enumerate(placements_list):
        frame_hr = frame_hr_batch[frame_idx]
        for entry in placements:
            if entry.bin_idx >= len(sr_bins):
                continue
            sr_bin = sr_bins[entry.bin_idx]
            sx1, sy1 = entry.bx * scale, entry.by * scale
            sx2, sy2 = sx1 + entry.bw * scale, sy1 + entry.bh * scale
            sr_patch = sr_bin[:, sy1:sy2, sx1:sx2]
            x, y, w, h = entry.orig_bbox
            dx1, dy1 = x * scale, y * scale
            dx2 = min(frame_hr.shape[2], (x + w) * scale)
            dy2 = min(frame_hr.shape[1], (y + h) * scale)
            dw, dh = dx2 - dx1, dy2 - dy1
            if dw <= 0 or dh <= 0 or sr_patch.numel() == 0:
                continue
            if sr_patch.shape[1] != dh or sr_patch.shape[2] != dw:
                started = time.time()
                sr_patch = F.interpolate(
                    sr_patch.unsqueeze(0),
                    size=(dh, dw),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                timing["patch_resize_seconds"] += time.time() - started
            started = time.time()
            frame_hr[:, dy1:dy2, dx1:dx2] = sr_patch
            timing["patch_paste_seconds"] += time.time() - started
    frame_hrs = list(frame_hr_batch)
    if return_timing:
        return frame_hrs, timing
    return frame_hrs
