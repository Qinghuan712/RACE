from __future__ import annotations

from pathlib import Path
from typing import Any
from contextlib import contextmanager

import pycuda.driver as cuda
import tensorrt as trt
import torch


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


class SRBatchInfer:
    """TensorRT SR wrapper that executes a full CUDA NCHW batch in one enqueue."""

    def __init__(self, args: Any, cfx: cuda.Context, *, scale: int = 3):
        self.model_path = Path(args.sr_model_path)
        self.cfx = cfx
        self.scale = scale
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.engine = self.runtime.deserialize_cuda_engine(self.model_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.model_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(
                f"Failed to create TensorRT execution context for {self.model_path}. "
                "This usually means GPU memory is insufficient for this engine/profile."
            )
        self.stream = cuda.Stream()
        self.input_name = self._tensor_name(0)
        self.output_name = self._tensor_name(1)

    def _tensor_name(self, index: int) -> str:
        if hasattr(self.engine, "get_tensor_name"):
            return self.engine.get_tensor_name(index)
        return self.engine.get_binding_name(index)

    def _set_input_shape(self, shape: tuple[int, int, int, int]) -> None:
        """Set dynamic TRT input shape, supporting TRT 8 and TRT 10 APIs."""

        try:
            ok = self.context.set_input_shape(self.input_name, shape)
            if ok is False:
                raise RuntimeError("TensorRT rejected input shape")
        except AttributeError:
            ok = self.context.set_binding_shape(0, shape)
            if ok is False:
                raise RuntimeError("TensorRT rejected binding shape")
        except Exception as exc:
            raise RuntimeError(
                f"SR engine does not support input shape {shape}. "
                "Rebuild the TensorRT engine with a dynamic batch profile that covers this K, "
                "for example via `python RACE/build_dynamic_sr_engine.py --fp16`."
            ) from exc

    def _get_output_shape(self, batch: int, height: int, width: int) -> tuple[int, int, int, int]:
        """Resolve TRT output shape, falling back to the configured SR scale."""

        shape = None
        try:
            shape = tuple(int(dim) for dim in self.context.get_tensor_shape(self.output_name))
        except AttributeError:
            try:
                shape = tuple(int(dim) for dim in self.context.get_binding_shape(1))
            except Exception:
                shape = None

        if shape and len(shape) == 4 and all(dim > 0 for dim in shape):
            return shape
        return (batch, 3, height * self.scale, width * self.scale)

    def inference(self, imgs: torch.Tensor) -> torch.Tensor:
        """Run SR on an already-GPU tensor and return a GPU tensor output."""

        if imgs.ndim != 4:
            raise ValueError(f"Expected NCHW input tensor, got shape={tuple(imgs.shape)}")
        if not imgs.is_cuda:
            raise ValueError("SRBatchInfer expects a CUDA tensor input")

        imgs = imgs.contiguous()
        batch, channels, height, width = (int(dim) for dim in imgs.shape)
        if channels != 3:
            raise ValueError(f"Expected 3-channel input, got {channels}")

        input_shape = (batch, channels, height, width)
        with nvtx_range(f"sr_setup:shape={input_shape}"):
            self._set_input_shape(input_shape)
            output_shape = self._get_output_shape(batch, height, width)
            if output_shape[0] != batch:
                raise RuntimeError(
                    f"SR engine output batch {output_shape[0]} does not match input batch {batch}. "
                    "Rebuild the TensorRT engine with a dynamic batch profile for true batch inference."
                )
            output = torch.empty(output_shape, device=imgs.device, dtype=imgs.dtype).contiguous()

        input_ptr = int(imgs.data_ptr())
        output_ptr = int(output.data_ptr())

        if hasattr(self.context, "set_tensor_address"):
            self.context.set_tensor_address(self.input_name, input_ptr)
            self.context.set_tensor_address(self.output_name, output_ptr)
            self.cfx.push()
            try:
                with nvtx_range(f"sr_enqueue:shape={input_shape}"):
                    ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
            finally:
                self.cfx.pop()
        else:
            self.cfx.push()
            try:
                with nvtx_range(f"sr_enqueue:shape={input_shape}"):
                    ok = self.context.execute_async_v2(
                        bindings=[input_ptr, output_ptr],
                        stream_handle=self.stream.handle,
                    )
            finally:
                self.cfx.pop()

        if ok is False:
            raise RuntimeError(f"TensorRT execution failed for input shape {input_shape}")
        # Keep this synchronize explicit: timing summaries treat SR inference as
        # a completed stage before the blend/detect work starts.
        with nvtx_range(f"sr_sync:shape={input_shape}"):
            self.stream.synchronize()
        return output

    def close(self) -> None:
        self.cfx.push()
        try:
            if self.stream is not None:
                self.stream.synchronize()
            cuda.Context.synchronize()
            self.context = None
            self.engine = None
            self.runtime = None
            self.stream = None
        finally:
            self.cfx.pop()
