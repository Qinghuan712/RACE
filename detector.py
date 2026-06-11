"""
Yolo11TRTDetector — YOLO11/v8 TensorRT inference wrapper for RACE.

The runtime supports three engine families:
    1. raw_yolo: one raw output, e.g. (B, 84, anchors), parsed in Python.
    2. nms_single: Ultralytics NMS plugin output, e.g. (B, 300, 6).
    3. nms_multi: legacy YOLOv5 NMS plugin outputs: num/boxes/scores/labels.

RACE filters detections here before writing detections.json, using
conf_thresh=0.5 and COCO vehicle class ids (2, 5, 7) by default.

Usage example:
    import pycuda.driver as cuda
    cuda.init()
    cfx = cuda.Device(0).make_context()

    from RACE.detector import Yolo11TRTDetector
    detector = Yolo11TRTDetector("path/to/yolo11n.engine", cfx)

    # img_tensor: torch.Tensor, shape (3, H, W), float32, 0~255, on CUDA
    num, boxes, scores, labels = detector.inference_raw(img_tensor)
    cfx.pop()
"""

import numpy as np
import torch
import torch.nn.functional as F
import tensorrt as trt
import pycuda.driver as cuda
import time
from contextlib import contextmanager


DEFAULT_DETECT_CLASS_IDS = (2, 5, 7)


def _torch_dtype_from_trt(dtype) -> torch.dtype:
    if dtype == trt.float16:
        return torch.float16
    if dtype == trt.float32:
        return torch.float32
    if dtype == trt.int32:
        return torch.int32
    if hasattr(trt, "int64") and dtype == trt.int64:
        return torch.int64
    if dtype == trt.bool:
        return torch.bool
    raise TypeError(f"Unsupported TensorRT dtype: {dtype}")


def _normalize_class_ids(class_ids) -> tuple[int, ...] | None:
    """Normalize CLI/API class-id input into a sorted tuple or no filter."""

    if class_ids is None:
        return None
    if isinstance(class_ids, str):
        raw = class_ids.strip()
        if not raw or raw.lower() in {"all", "none", "*"}:
            return None
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    else:
        values = [int(item) for item in class_ids]
    return tuple(sorted(set(values)))


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


class Yolo11TRTDetector:
    """
    YOLO11/v8 TensorRT detector using TRT 10.x tensor APIs.

    Supports:
      - raw YOLO output requiring Python top-k parsing
      - Ultralytics NMS single output shaped (B, max_det, 6)
      - legacy NMS multi-output engines with num / boxes / scores / labels

    Automatically reads optimal input size from engine profile.
    """

    def __init__(
        self,
        engine_path: str,
        cfx,
        conf_thresh: float = 0.5,
        class_ids: tuple[int, ...] | list[int] | set[int] | str | None = DEFAULT_DETECT_CLASS_IDS,
    ):
        """
        Args:
            engine_path: Path to TensorRT .engine file
            cfx:         Active pycuda CUDA context
            conf_thresh: Detection confidence threshold
            class_ids:   COCO class ids to keep. None/all disables class filtering.
        """
        print(f"[Yolo11TRTDetector] Loading engine: {engine_path}")
        self.cfx = cfx
        self.conf_thresh = float(conf_thresh)
        self.class_ids = _normalize_class_ids(class_ids)

        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        self.runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger, "")
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.stream = cuda.Stream()
        self.context = self.engine.create_execution_context()
        self.last_timing: dict[str, float] = {}

        # ── Enumerate IO tensors ─────────────────────────────────────
        self.input_names = []
        self.output_names = []
        self.output_shapes = {}
        self.output_dtypes = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
                self.output_shapes[name] = shape
                self.output_dtypes[name] = dtype
            print(f"  tensor '{name}': mode={mode}, shape={shape}")

        # ── Read optimal input size from engine profile ──────────────
        inp_name = self.input_names[0]
        self.min_batch = 1
        self.opt_batch = 1
        self.max_batch = 1
        try:
            min_shape, opt_shape, max_shape = self.engine.get_tensor_profile_shape(inp_name, 0)
            print(f"  input profile: min={min_shape}, opt={opt_shape}, max={max_shape}")
            self.min_batch = int(min_shape[0])
            self.opt_batch = int(opt_shape[0])
            self.max_batch = int(max_shape[0])
            self.input_h = opt_shape[2]
            self.input_w = opt_shape[3]
        except Exception as e:
            print(f"  Cannot read profile shape ({e}), using default 640x640")
            self.input_h = 640
            self.input_w = 640

        self.context.set_input_shape(inp_name, (1, 3, self.input_h, self.input_w))
        print(f"  using input shape: (1, 3, {self.input_h}, {self.input_w})")

        # ── Detect engine type ───────────────────────────────────────
        # raw yolo11/v8: one output shaped roughly (B, 84, anchors)
        # Ultralytics NMS: one output shaped roughly (B, max_det, 6)
        # legacy yolov5+NMS: four outputs num / boxes / scores / labels
        self.engine_kind = self._detect_engine_kind()
        self.is_yolov5_nms = (self.engine_kind == "nms_multi")
        self.is_nms_engine = self.engine_kind in {"nms_single", "nms_multi"}
        self.nms_output_roles = self._resolve_nms_multi_output_roles() if self.is_yolov5_nms else {}
        print(
            f"  num_outputs={len(self.output_names)}, engine_kind={self.engine_kind}, "
            f"is_nms_engine={self.is_nms_engine}"
        )
        class_filter = "all" if self.class_ids is None else ",".join(str(class_id) for class_id in self.class_ids)
        print(f"  detector_filter: conf_thresh={self.conf_thresh:.3f}, class_ids={class_filter}")
        print("[Yolo11TRTDetector] Ready.")

    def close(self) -> None:
        self.cfx.push()
        try:
            if getattr(self, "stream", None) is not None:
                self.stream.synchronize()
            cuda.Context.synchronize()
            self.context = None
            self.engine = None
            self.runtime = None
            self.stream = None
        finally:
            self.cfx.pop()

    # ─────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────

    def inference_raw(self, img_tensor: torch.Tensor):
        """
        Run TRT detection inference on a single image.

        Args:
            img_tensor: torch.Tensor, shape (3, H, W), values 0~255, on CUDA

        Returns:
            num_arr    (np.ndarray, shape (1,)):  number of detections
            boxes_arr  (np.ndarray, shape (N,4)): [x1, y1, x2, y2] in original image space
            scores_arr (np.ndarray, shape (N,)):  confidence scores
            labels_arr (np.ndarray, shape (N,)):  class ids
        """
        C, orig_h, orig_w = img_tensor.shape
        img = img_tensor.unsqueeze(0).float()  # (1, 3, H, W)
        img = F.interpolate(img, size=(self.input_h, self.input_w),
                            mode="bilinear", align_corners=False)
        img = img / 255.0

        if self.engine_kind == "nms_single":
            nms_output = self._inference_nms_single_batch(img)
            boxes_arr, scores_arr, labels_arr = self._parse_nms_single_output_torch_batch(nms_output, 1.0, 1.0)[0]
            num_arr = np.array([len(scores_arr)], dtype=np.int32)
        elif self.engine_kind == "nms_multi":
            nms_outputs = self._inference_nms_multi_batch(img)
            boxes_arr, scores_arr, labels_arr = self._parse_nms_multi_output_torch_batch(nms_outputs, 1.0, 1.0)[0]
            num_arr = np.array([len(scores_arr)], dtype=np.int32)
        else:
            num_arr, boxes_arr, scores_arr, labels_arr = self._inference_yolo11(img)

        # Scale box coordinates back to original image space
        if len(boxes_arr) > 0:
            scale_x = orig_w / self.input_w
            scale_y = orig_h / self.input_h
            boxes_arr[:, 0] *= scale_x
            boxes_arr[:, 1] *= scale_y
            boxes_arr[:, 2] *= scale_x
            boxes_arr[:, 3] *= scale_y

        return num_arr, boxes_arr, scores_arr, labels_arr

    def inference_raw_batch(self, img_batch: torch.Tensor):
        """
        Run TRT detection on a batch of images.

        Args:
            img_batch: torch.Tensor, shape (N, 3, H, W), values 0~255, on CUDA

        Returns:
            List of (boxes, scores, labels), one tuple per image.
        """
        if img_batch.ndim != 4:
            raise ValueError(f"Expected NCHW batch tensor, got shape {tuple(img_batch.shape)}")
        batch, channels, orig_h, orig_w = (int(dim) for dim in img_batch.shape)
        if channels != 3:
            raise ValueError(f"Expected 3 channels, got {channels}")
        if batch <= 0:
            return []

        # Batch-1 engines fall back to single-image inference.
        if self.max_batch <= 1:
            return [
                self.inference_raw(img_batch[idx])[1:]
                for idx in range(batch)
            ]
        if batch > self.max_batch:
            raise ValueError(
                f"Detector batch {batch} exceeds engine max batch {self.max_batch}. "
                "Split the workload into smaller chunks."
            )

        started = time.time()
        with nvtx_range(f"detector_preprocess:batch={batch}"):
            img = img_batch.float()
            img = F.interpolate(img, size=(self.input_h, self.input_w), mode="bilinear", align_corners=False)
            img = img / 255.0
        preprocess_seconds = time.time() - started

        infer_started = time.time()
        with nvtx_range(f"detector_trt:batch={batch}"):
            if self.engine_kind == "nms_single":
                raw, trt_detail = self._inference_nms_single_batch(img, return_timing=True)
            elif self.engine_kind == "nms_multi":
                raw, trt_detail = self._inference_nms_multi_batch(img, return_timing=True)
            else:
                raw, trt_detail = self._inference_yolo11_batch(img, return_timing=True)
        trt_seconds = time.time() - infer_started
        first_output = next(iter(raw.values())) if isinstance(raw, dict) else raw
        if int(first_output.shape[0]) != batch:
            raise RuntimeError(
                f"Detector output batch {first_output.shape[0]} does not match input batch {batch}"
            )

        parse_started = time.time()
        with nvtx_range(f"detector_parse:batch={batch}"):
            scale_x = orig_w / self.input_w
            scale_y = orig_h / self.input_h
            # NMS engines already suppress boxes inside TRT. Raw YOLO engines
            # still need lightweight score/class filtering in Python.
            if self.engine_kind == "nms_single":
                outputs = self._parse_nms_single_output_torch_batch(raw, scale_x, scale_y)
            elif self.engine_kind == "nms_multi":
                outputs = self._parse_nms_multi_output_torch_batch(raw, scale_x, scale_y)
            else:
                outputs = self._parse_yolo_output_torch_batch(raw, scale_x, scale_y)
        parse_seconds = time.time() - parse_started
        self.last_timing = {
            "preprocess_seconds": preprocess_seconds,
            "trt_seconds": trt_seconds,
            "trt_setup_seconds": float(trt_detail.get("setup_seconds", 0.0)),
            "trt_enqueue_seconds": float(trt_detail.get("enqueue_seconds", 0.0)),
            "trt_sync_seconds": float(trt_detail.get("sync_seconds", 0.0)),
            "parse_seconds": parse_seconds,
        }
        return outputs

    # ─────────────────────────────────────────────────────────────────
    # Private inference methods
    # ─────────────────────────────────────────────────────────────────

    def _detect_engine_kind(self) -> str:
        """Infer which parser to use from TensorRT output count and shape."""

        if len(self.output_names) == 4:
            return "nms_multi"
        if len(self.output_names) == 1:
            shape = self.output_shapes[self.output_names[0]]
            if len(shape) >= 2 and int(shape[-1]) == 6:
                return "nms_single"
        return "raw_yolo"

    def _resolve_nms_multi_output_roles(self) -> dict[str, str]:
        """Map generic TRT output names to num/boxes/scores/labels roles."""

        roles: dict[str, str] = {}
        for name in self.output_names:
            lower = name.lower()
            shape = self.output_shapes[name]
            dtype = self.output_dtypes.get(name)
            if "num" in lower:
                roles["num"] = name
            elif "box" in lower or (len(shape) >= 2 and int(shape[-1]) == 4):
                roles["boxes"] = name
            elif "score" in lower or "conf" in lower:
                roles["scores"] = name
            elif "label" in lower or "class" in lower:
                roles["labels"] = name
            elif dtype == trt.int32:
                roles.setdefault("labels", name)

        # Preserve compatibility with older engines whose output names are generic.
        fallback = ["num", "boxes", "scores", "labels"]
        for role, name in zip(fallback, self.output_names):
            roles.setdefault(role, name)

        missing = [role for role in fallback if role not in roles]
        if missing:
            raise RuntimeError(f"Could not resolve NMS output roles, missing: {missing}")
        print(f"  nms output roles: {roles}")
        return roles

    def _resolved_output_shape(self, name: str, batch: int) -> tuple[int, ...]:
        """Resolve dynamic output dimensions after setting the input shape."""

        shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
        if not shape:
            shape = tuple(int(dim) for dim in self.output_shapes[name])
        resolved = []
        for idx, dim in enumerate(shape):
            if dim < 0:
                if idx == 0:
                    resolved.append(batch)
                else:
                    static_dim = int(self.output_shapes[name][idx])
                    if static_dim < 0:
                        raise RuntimeError(f"Output shape for {name} is still dynamic: {shape}")
                    resolved.append(static_dim)
            else:
                resolved.append(dim)
        return tuple(resolved)

    def _allocate_output_tensor(self, name: str, batch: int, device: torch.device) -> torch.Tensor:
        shape = self._resolved_output_shape(name, batch)
        dtype = _torch_dtype_from_trt(self.output_dtypes[name])
        return torch.empty(*shape, dtype=dtype, device=device)

    def _class_keep_torch(self, labels: torch.Tensor) -> torch.Tensor:
        """Return a GPU boolean mask for the configured class filter."""

        if self.class_ids is None:
            return torch.ones_like(labels, dtype=torch.bool)
        allowed = torch.tensor(self.class_ids, dtype=labels.dtype, device=labels.device)
        return (labels.unsqueeze(-1) == allowed).any(dim=-1)

    def _class_keep_numpy(self, labels: np.ndarray) -> np.ndarray:
        """Return a NumPy boolean mask for the configured class filter."""

        if self.class_ids is None:
            return np.ones_like(labels, dtype=bool)
        return np.isin(labels, np.asarray(self.class_ids, dtype=labels.dtype))

    def _inference_yolov5_nms(self, img: torch.Tensor):
        """yolov5+NMS engine: 4 outputs (num, boxes, scores, labels)"""
        outputs = self._inference_nms_multi_batch(img)
        boxes, scores, labels = self._parse_nms_multi_output_torch_batch(outputs, 1.0, 1.0)[0]
        return (
            np.array([len(scores)], dtype=np.int32),
            boxes,
            scores,
            labels,
        )

    def _inference_yolo11(self, img: torch.Tensor):
        """yolo11/v8 engine: single output, requires post-processing NMS"""
        out_name = self.output_names[0]
        out_shape = self.context.get_tensor_shape(out_name)
        out_tensor = torch.empty(*out_shape, dtype=torch.float32).cuda()

        self.cfx.push()
        self.context.set_tensor_address(self.input_names[0], int(img.data_ptr()))
        self.context.set_tensor_address(out_name, int(out_tensor.data_ptr()))
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        self.cfx.pop()
        self.stream.synchronize()

        raw = out_tensor.cpu().numpy()
        return self._parse_yolo_output(raw)

    def _inference_yolo11_batch(
        self,
        img: torch.Tensor,
        *,
        return_timing: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Batched yolo11/v8 inference on a normalized NCHW CUDA tensor."""
        input_name = self.input_names[0]
        out_name = self.output_names[0]
        batch = int(img.shape[0])
        timing = {
            "setup_seconds": 0.0,
            "enqueue_seconds": 0.0,
            "sync_seconds": 0.0,
        }

        self.cfx.push()
        try:
            with nvtx_range(f"detector_trt_setup:batch={batch}"):
                started = time.time()
                self.context.set_input_shape(input_name, tuple(int(dim) for dim in img.shape))
                out_shape = tuple(int(dim) for dim in self.context.get_tensor_shape(out_name))
                if out_shape and out_shape[0] < 0:
                    out_shape = (batch, *out_shape[1:])
                out_tensor = torch.empty(*out_shape, dtype=torch.float32, device=img.device)
                self.context.set_tensor_address(input_name, int(img.data_ptr()))
                self.context.set_tensor_address(out_name, int(out_tensor.data_ptr()))
                timing["setup_seconds"] = time.time() - started
            with nvtx_range(f"detector_trt_enqueue:batch={batch}"):
                started = time.time()
                self.context.execute_async_v3(stream_handle=self.stream.handle)
                timing["enqueue_seconds"] = time.time() - started
        finally:
            self.cfx.pop()
        with nvtx_range(f"detector_trt_sync:batch={batch}"):
            started = time.time()
            self.stream.synchronize()
            timing["sync_seconds"] = time.time() - started
        if return_timing:
            return out_tensor, timing
        return out_tensor

    def _inference_nms_single_batch(
        self,
        img: torch.Tensor,
        *,
        return_timing: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Batched Ultralytics NMS inference, output shaped (B, max_det, 6)."""
        input_name = self.input_names[0]
        out_name = self.output_names[0]
        batch = int(img.shape[0])
        timing = {
            "setup_seconds": 0.0,
            "enqueue_seconds": 0.0,
            "sync_seconds": 0.0,
        }

        self.cfx.push()
        try:
            with nvtx_range(f"detector_trt_setup:nms_single_batch={batch}"):
                started = time.time()
                self.context.set_input_shape(input_name, tuple(int(dim) for dim in img.shape))
                out_tensor = self._allocate_output_tensor(out_name, batch, img.device)
                self.context.set_tensor_address(input_name, int(img.data_ptr()))
                self.context.set_tensor_address(out_name, int(out_tensor.data_ptr()))
                timing["setup_seconds"] = time.time() - started
            with nvtx_range(f"detector_trt_enqueue:nms_single_batch={batch}"):
                started = time.time()
                self.context.execute_async_v3(stream_handle=self.stream.handle)
                timing["enqueue_seconds"] = time.time() - started
        finally:
            self.cfx.pop()
        with nvtx_range(f"detector_trt_sync:nms_single_batch={batch}"):
            started = time.time()
            self.stream.synchronize()
            timing["sync_seconds"] = time.time() - started
        if return_timing:
            return out_tensor, timing
        return out_tensor

    def _inference_nms_multi_batch(
        self,
        img: torch.Tensor,
        *,
        return_timing: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Batched legacy NMS inference with num / boxes / scores / labels outputs."""
        input_name = self.input_names[0]
        batch = int(img.shape[0])
        timing = {
            "setup_seconds": 0.0,
            "enqueue_seconds": 0.0,
            "sync_seconds": 0.0,
        }

        self.cfx.push()
        try:
            with nvtx_range(f"detector_trt_setup:nms_multi_batch={batch}"):
                started = time.time()
                self.context.set_input_shape(input_name, tuple(int(dim) for dim in img.shape))
                outputs = {
                    name: self._allocate_output_tensor(name, batch, img.device)
                    for name in self.output_names
                }
                self.context.set_tensor_address(input_name, int(img.data_ptr()))
                for name, tensor in outputs.items():
                    self.context.set_tensor_address(name, int(tensor.data_ptr()))
                timing["setup_seconds"] = time.time() - started
            with nvtx_range(f"detector_trt_enqueue:nms_multi_batch={batch}"):
                started = time.time()
                self.context.execute_async_v3(stream_handle=self.stream.handle)
                timing["enqueue_seconds"] = time.time() - started
        finally:
            self.cfx.pop()
        with nvtx_range(f"detector_trt_sync:nms_multi_batch={batch}"):
            started = time.time()
            self.stream.synchronize()
            timing["sync_seconds"] = time.time() - started
        if return_timing:
            return outputs, timing
        return outputs

    def _parse_nms_single_output_torch_batch(
        self,
        nms_output: torch.Tensor,
        scale_x: float,
        scale_y: float,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Parse Ultralytics NMS output: (B, max_det, 6), columns are
        x1, y1, x2, y2, score, class in detector input image space.
        """
        if nms_output.ndim == 2:
            nms_output = nms_output.unsqueeze(0)
        if nms_output.ndim != 3 or int(nms_output.shape[-1]) < 6:
            raise ValueError(f"Expected NMS output (B, max_det, 6), got {tuple(nms_output.shape)}")

        dets = nms_output[..., :6].float()
        boxes = dets[..., :4].clone()
        boxes[..., 0] *= scale_x
        boxes[..., 2] *= scale_x
        boxes[..., 1] *= scale_y
        boxes[..., 3] *= scale_y
        scores = dets[..., 4]
        labels = dets[..., 5].to(torch.int32)
        valid = (scores >= float(self.conf_thresh)) & self._class_keep_torch(labels)

        boxes_np = boxes.detach().cpu().numpy().astype(np.float32, copy=False)
        scores_np = scores.detach().cpu().numpy().astype(np.float32, copy=False)
        labels_np = labels.detach().cpu().numpy().astype(np.int32, copy=False)
        valid_np = valid.detach().cpu().numpy()

        outputs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for idx in range(int(boxes_np.shape[0])):
            keep = valid_np[idx]
            if not bool(keep.any()):
                outputs.append((
                    np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.int32),
                ))
                continue
            outputs.append((
                boxes_np[idx][keep],
                scores_np[idx][keep],
                labels_np[idx][keep],
            ))
        return outputs

    def _parse_nms_multi_output_torch_batch(
        self,
        outputs_by_name: dict[str, torch.Tensor],
        scale_x: float,
        scale_y: float,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Parse NMS plugin outputs that expose num/boxes/scores/labels."""

        roles = self.nms_output_roles
        num = outputs_by_name[roles["num"]].reshape(-1).to(torch.int32)
        boxes = outputs_by_name[roles["boxes"]].float()
        scores = outputs_by_name[roles["scores"]].float()
        labels = outputs_by_name[roles["labels"]].to(torch.int32)
        if boxes.ndim == 2:
            boxes = boxes.unsqueeze(0)
        if scores.ndim == 1:
            scores = scores.unsqueeze(0)
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)

        boxes = boxes.clone()
        boxes[..., 0] *= scale_x
        boxes[..., 2] *= scale_x
        boxes[..., 1] *= scale_y
        boxes[..., 3] *= scale_y

        boxes_np = boxes.detach().cpu().numpy().astype(np.float32, copy=False)
        scores_np = scores.detach().cpu().numpy().astype(np.float32, copy=False)
        labels_np = labels.detach().cpu().numpy().astype(np.int32, copy=False)
        num_np = num.detach().cpu().numpy().astype(np.int32, copy=False)

        outputs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for idx in range(int(boxes_np.shape[0])):
            count = int(num_np[min(idx, len(num_np) - 1)])
            count = max(0, min(count, int(boxes_np.shape[1])))
            if count <= 0:
                outputs.append((
                    np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.int32),
                ))
                continue
            valid = (
                (scores_np[idx, :count] >= float(self.conf_thresh))
                & self._class_keep_numpy(labels_np[idx, :count])
            )
            outputs.append((
                boxes_np[idx, :count][valid],
                scores_np[idx, :count][valid],
                labels_np[idx, :count][valid],
            ))
        return outputs

    def _parse_yolo_output_torch_batch(
        self,
        raw: torch.Tensor,
        scale_x: float,
        scale_y: float,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Parse batched yolo v8/v11 raw output on torch tensors.
        Keep the heavy selection regular at B x topK, then copy compact
        tensors back to CPU for final per-image slicing.
        """
        if raw.ndim != 3:
            raise ValueError(f"Expected batched YOLO output with 3 dims, got {tuple(raw.shape)}")
        if raw.shape[1] < raw.shape[2]:
            raw = raw.transpose(1, 2).contiguous()  # (B, C, N) -> (B, N, C)

        boxes_cxcywh = raw[..., :4]
        class_scores = raw[..., 4:]
        max_scores, max_labels = class_scores.max(dim=-1)
        topk_count = min(100, int(max_scores.shape[1]))
        top_scores, top_indices = max_scores.topk(topk_count, dim=1)
        top_labels = max_labels.gather(1, top_indices)
        top_boxes = boxes_cxcywh.gather(
            1,
            top_indices.unsqueeze(-1).expand(-1, -1, boxes_cxcywh.shape[-1]),
        )

        cx, cy, w, h = top_boxes.unbind(dim=2)
        top_xyxy = torch.stack(
            (
                (cx - w * 0.5) * scale_x,
                (cy - h * 0.5) * scale_y,
                (cx + w * 0.5) * scale_x,
                (cy + h * 0.5) * scale_y,
            ),
            dim=2,
        )
        valid = (top_scores >= float(self.conf_thresh)) & self._class_keep_torch(top_labels)

        boxes_np = top_xyxy.detach().cpu().numpy().astype(np.float32, copy=False)
        scores_np = top_scores.detach().cpu().numpy().astype(np.float32, copy=False)
        labels_np = top_labels.detach().cpu().numpy().astype(np.int32, copy=False)
        valid_np = valid.detach().cpu().numpy()

        outputs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for idx in range(int(boxes_np.shape[0])):
            keep = valid_np[idx]
            if not bool(keep.any()):
                outputs.append((
                    np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.int32),
                ))
                continue
            outputs.append((
                boxes_np[idx][keep],
                scores_np[idx][keep],
                labels_np[idx][keep],
            ))
        return outputs

    def _parse_yolo_output(self, raw: np.ndarray):
        """
        Parse yolo v8/v11 raw output.
        raw shape: (1, 84, 8400) or (1, 8400, 84)
        84 = 4 (cx, cy, w, h) + 80 class scores
        """
        raw = raw.squeeze(0)
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T  # (84, N) -> (N, 84)

        cx, cy, w, h = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
        class_scores = raw[:, 4:]
        max_scores = class_scores.max(axis=1)
        max_labels = class_scores.argmax(axis=1)

        mask = (max_scores >= self.conf_thresh) & self._class_keep_numpy(max_labels)
        cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
        max_scores = max_scores[mask]
        max_labels = max_labels[mask]

        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2

        num_dets = len(max_scores)
        if num_dets > 100:
            topk = np.argsort(max_scores)[::-1][:100]
            x1, y1, x2, y2 = x1[topk], y1[topk], x2[topk], y2[topk]
            max_scores, max_labels = max_scores[topk], max_labels[topk]
            num_dets = 100

        boxes_arr = (np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
                     if num_dets > 0 else np.zeros((0, 4), dtype=np.float32))
        return (np.array([num_dets], dtype=np.int32),
                boxes_arr,
                max_scores.astype(np.float32),
                max_labels.astype(np.int32))
