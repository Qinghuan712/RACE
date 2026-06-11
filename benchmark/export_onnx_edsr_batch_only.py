from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import torch


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "model" / "edsr.py").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate repo root from {start}; expected to find model/edsr.py in a parent directory."
    )


REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import edsr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export EDSR to ONNX with batch-only dynamic axis and fixed spatial size."
    )
    parser.add_argument("--weights", default="temp_model/EDSR_x3.pt")
    parser.add_argument("--output", default="temp_model/EDSR_x3_batch_dynamic.onnx")
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--opset", type=int, default=11)
    parser.add_argument("--n_resblocks", type=int, default=32)
    parser.add_argument("--n_feats", type=int, default=256)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--rgb_range", type=int, default=255)
    parser.add_argument("--n_colors", type=int, default=3)
    parser.add_argument("--res_scale", type=float, default=0.1)
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    model_args = types.SimpleNamespace()
    model_args.n_resblocks = args.n_resblocks
    model_args.n_feats = args.n_feats
    model_args.scale = [args.scale]
    model_args.rgb_range = args.rgb_range
    model_args.n_colors = args.n_colors
    model_args.res_scale = args.res_scale

    model = edsr.EDSR(model_args)
    weights_path = REPO_ROOT / args.weights
    if not weights_path.exists():
        raise FileNotFoundError(f"EDSR weights not found: {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    model = build_model(args)

    dummy_input = torch.randn(1, args.n_colors, args.height, args.width)
    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=args.opset,
        do_constant_folding=True,
        export_params=True,
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"Exported batch-dynamic ONNX to {output_path}")
    print(f"Input shape profile: [N, {args.n_colors}, {args.height}, {args.width}]")


if __name__ == "__main__":
    main()
