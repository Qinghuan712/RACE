from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


DEFAULT_TRTEXEC_CANDIDATES = [
    "trtexec",
    "./TensorRT-8.6.1.6/targets/x86_64-linux-gnu/bin/trtexec",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a dynamic-batch TensorRT engine from ONNX.")
    parser.add_argument("--onnx", default="temp_model/EDSR_x3.onnx")
    parser.add_argument("--output_engine", default="temp_model/EDSR_x3_b1_b8.engine")
    parser.add_argument(
        "--input_name",
        default=None,
        help="Optional input tensor name. When omitted, the script tries to infer it from the ONNX graph.",
    )
    parser.add_argument("--min_batch", type=int, default=1)
    parser.add_argument("--opt_batch", type=int, default=4)
    parser.add_argument("--max_batch", type=int, default=8)
    parser.add_argument("--min_height", type=int, default=360)
    parser.add_argument("--opt_height", type=int, default=360)
    parser.add_argument("--max_height", type=int, default=360)
    parser.add_argument("--min_width", type=int, default=640)
    parser.add_argument("--opt_width", type=int, default=640)
    parser.add_argument("--max_width", type=int, default=640)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--workspace_mib", type=int, default=4096)
    parser.add_argument(
        "--backend",
        choices=("auto", "trtexec", "python"),
        default="auto",
        help="How to build the TensorRT engine. `auto` prefers trtexec and falls back to TensorRT Python API.",
    )
    parser.add_argument("--trtexec", default=None, help="Optional explicit path to trtexec.")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def find_trtexec(explicit: str | None) -> str:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"trtexec not found: {explicit}")
        if not path.is_file():
            raise FileNotFoundError(f"trtexec path is not a file: {explicit}")
        if not os.access(path, os.X_OK):
            raise PermissionError(f"trtexec is not executable: {explicit}")
        return str(path)

    for candidate in DEFAULT_TRTEXEC_CANDIDATES:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        path = Path(candidate)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    raise FileNotFoundError(
        "Could not locate trtexec. Pass --trtexec or add it to PATH."
    )


def shape_spec(name: str, batch: int, height: int, width: int) -> str:
    return f"{name}:{batch}x3x{height}x{width}"


def _inspect_network_input_names(onnx_path: Path) -> list[str]:
    try:
        import tensorrt as trt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorRT Python module is required to inspect ONNX input names automatically. "
            "Pass --input_name explicitly if you do not have tensorrt Python installed."
        ) from exc

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)

    parsed = False
    if hasattr(parser, "parse_from_file"):
        parsed = bool(parser.parse_from_file(str(onnx_path)))
    if not parsed:
        parsed = bool(parser.parse(onnx_path.read_bytes()))
    if not parsed:
        errors = [str(parser.get_error(idx)) for idx in range(parser.num_errors)]
        raise RuntimeError(
            "Failed to parse ONNX model while inspecting input names:\n" + "\n".join(errors)
        )
    return [network.get_input(i).name for i in range(network.num_inputs)]


def _resolve_onnx_input_name(onnx_path: Path, requested_name: str | None) -> str:
    input_names = _inspect_network_input_names(onnx_path)
    if requested_name and requested_name in input_names:
        return requested_name
    if requested_name and len(input_names) > 1:
        raise ValueError(
            f"Could not resolve input name `{requested_name}`. Available network inputs: {input_names}"
        )
    if len(input_names) == 1:
        actual = input_names[0]
        if requested_name and requested_name != actual:
            print(f"Requested input name `{requested_name}` not found; using network input `{actual}` instead.")
        elif requested_name is None:
            print(f"Auto-detected network input `{actual}`.")
        return actual
    if requested_name:
        return requested_name
    raise ValueError(
        "Could not infer input name automatically because the ONNX graph has multiple inputs. "
        f"Available inputs: {input_names}. Please pass --input_name explicitly."
    )


def build_with_trtexec(args: argparse.Namespace, onnx_path: Path, output_engine: Path) -> None:
    trtexec_path = find_trtexec(args.trtexec)
    input_name = _resolve_onnx_input_name(onnx_path, args.input_name)
    cmd = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={output_engine}",
        f"--minShapes={shape_spec(input_name, args.min_batch, args.min_height, args.min_width)}",
        f"--optShapes={shape_spec(input_name, args.opt_batch, args.opt_height, args.opt_width)}",
        f"--maxShapes={shape_spec(input_name, args.max_batch, args.max_height, args.max_width)}",
        f"--memPoolSize=workspace:{args.workspace_mib}",
    ]
    if args.fp16:
        cmd.append("--fp16")

    print("TensorRT build backend: trtexec")
    print("TensorRT build command:")
    print(" ".join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, check=True)


def _resolve_network_input_name(network, requested_name: str | None) -> str:
    input_names = [network.get_input(i).name for i in range(network.num_inputs)]
    if requested_name in input_names:
        return requested_name
    if len(input_names) == 1:
        actual = input_names[0]
        if requested_name:
            print(f"Requested input name `{requested_name}` not found; using network input `{actual}` instead.")
        else:
            print(f"Auto-detected network input `{actual}`.")
        return actual
    raise ValueError(
        f"Could not resolve input name `{requested_name}`. Available network inputs: {input_names}"
    )


def build_with_python(args: argparse.Namespace, onnx_path: Path, output_engine: Path) -> None:
    try:
        import tensorrt as trt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorRT Python module is not available in the current environment, "
            "so the Python builder backend cannot run."
        ) from exc

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)

    # Prefer parsing from file so TensorRT can resolve ONNX external data
    # such as `model.onnx.data` produced by some exporters.
    parsed = False
    if hasattr(parser, "parse_from_file"):
        parsed = bool(parser.parse_from_file(str(onnx_path)))
    if not parsed:
        parsed = bool(parser.parse(onnx_path.read_bytes()))

    if not parsed:
        errors = []
        for idx in range(parser.num_errors):
            errors.append(str(parser.get_error(idx)))
        raise RuntimeError(
            "Failed to parse ONNX model with TensorRT parser:\n" + "\n".join(errors)
        )

    input_name = _resolve_network_input_name(network, args.input_name)
    config = builder.create_builder_config()
    workspace_bytes = int(args.workspace_mib) * 1024 * 1024

    if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes

    if args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_name,
        (args.min_batch, 3, args.min_height, args.min_width),
        (args.opt_batch, 3, args.opt_height, args.opt_width),
        (args.max_batch, 3, args.max_height, args.max_width),
    )
    config.add_optimization_profile(profile)

    print("TensorRT build backend: python")
    print(
        "Optimization profile:",
        shape_spec(input_name, args.min_batch, args.min_height, args.min_width),
        shape_spec(input_name, args.opt_batch, args.opt_height, args.opt_width),
        shape_spec(input_name, args.max_batch, args.max_height, args.max_width),
    )
    if args.dry_run:
        return

    serialized = None
    if hasattr(builder, "build_serialized_network"):
        serialized = builder.build_serialized_network(network, config)
    else:
        engine = builder.build_engine(network, config)
        if engine is not None:
            serialized = engine.serialize()

    if serialized is None:
        raise RuntimeError("TensorRT builder failed to create a serialized engine.")

    output_engine.write_bytes(bytes(serialized))


def main() -> None:
    args = parse_args()
    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    if not (args.min_batch <= args.opt_batch <= args.max_batch):
        raise ValueError("Expected min_batch <= opt_batch <= max_batch")
    if not (args.min_height <= args.opt_height <= args.max_height):
        raise ValueError("Expected min_height <= opt_height <= max_height")
    if not (args.min_width <= args.opt_width <= args.max_width):
        raise ValueError("Expected min_width <= opt_width <= max_width")

    output_engine = Path(args.output_engine)
    output_engine.parent.mkdir(parents=True, exist_ok=True)
    if args.backend == "trtexec":
        build_with_trtexec(args, onnx_path, output_engine)
    elif args.backend == "python":
        build_with_python(args, onnx_path, output_engine)
    else:
        try:
            build_with_trtexec(args, onnx_path, output_engine)
        except (FileNotFoundError, PermissionError):
            print("Falling back to TensorRT Python builder because trtexec is unavailable.")
            build_with_python(args, onnx_path, output_engine)
    print(f"Saved engine: {output_engine}")


if __name__ == "__main__":
    main()
