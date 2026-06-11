from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite an ONNX model so batch and optional spatial dimensions are dynamic."
    )
    parser.add_argument("--input", required=True, help="Path to the source ONNX model.")
    parser.add_argument("--output", required=True, help="Path to the rewritten ONNX model.")
    parser.add_argument("--batch_symbol", default="batch_size")
    parser.add_argument("--dynamic_hw", action="store_true", help="Also rewrite H/W to dynamic symbols.")
    parser.add_argument("--height_symbol", default="height")
    parser.add_argument("--width_symbol", default="width")
    parser.add_argument(
        "--skip_check",
        action="store_true",
        help="Skip onnx.checker.check_model after rewriting.",
    )
    parser.add_argument(
        "--skip_infer_shapes",
        action="store_true",
        help="Skip shape inference before saving.",
    )
    return parser.parse_args()


def _set_dynamic_dims(value_info, batch_symbol: str, dynamic_hw: bool, height_symbol: str, width_symbol: str) -> bool:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return False
    dims = tensor_type.shape.dim
    if len(dims) == 0:
        return False
    dims[0].ClearField("dim_value")
    dims[0].dim_param = batch_symbol
    if dynamic_hw and len(dims) >= 4:
        dims[2].ClearField("dim_value")
        dims[2].dim_param = height_symbol
        dims[3].ClearField("dim_value")
        dims[3].dim_param = width_symbol
    return True


def main() -> None:
    args = parse_args()

    try:
        import onnx
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This script requires the `onnx` Python package. Install it in the current environment first."
        ) from exc

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {input_path}")

    model = onnx.load(str(input_path), load_external_data=True)

    touched = 0
    for value_info in list(model.graph.input) + list(model.graph.output):
        if _set_dynamic_dims(
            value_info,
            batch_symbol=args.batch_symbol,
            dynamic_hw=args.dynamic_hw,
            height_symbol=args.height_symbol,
            width_symbol=args.width_symbol,
        ):
            touched += 1

    if touched == 0:
        raise RuntimeError("Did not find any ONNX input/output tensors with shape information to rewrite.")

    if not args.skip_infer_shapes:
        try:
            model = onnx.shape_inference.infer_shapes(model)
        except Exception as exc:
            print(f"Warning: ONNX shape inference failed, continuing with rewritten model. Error: {exc}")

    if not args.skip_check:
        onnx.checker.check_model(model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=False,
    )

    if args.dynamic_hw:
        print(
            f"Rewrote dynamic dims to batch=`{args.batch_symbol}`, "
            f"height=`{args.height_symbol}`, width=`{args.width_symbol}`"
        )
    else:
        print(f"Rewrote batch dimension to `{args.batch_symbol}`")
    print(f"Input ONNX : {input_path}")
    print(f"Output ONNX: {output_path}")


if __name__ == "__main__":
    main()
