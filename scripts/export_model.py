#!/usr/bin/env python3
"""Export trained model to ONNX and TorchScript formats.

Usage:
    python scripts/export_model.py --checkpoint checkpoints/best_model.pt
    python scripts/export_model.py --checkpoint checkpoints/best_model.pt --format all
"""

import argparse
import time
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import DepthAwareStyleTransferNet, StyleTransferNet
from quantization.modelopt_utils import build_inference_model, export_inference_onnx


def export_onnx(
    model,
    output_path,
    image_size=256,
    opset=18,
    dynamic_axes=True,
    legacy=False,
    external_data=True,
    explicit_depth=False,
):
    print(f"Exporting ONNX to {output_path}...")
    export_inference_onnx(
        model,
        output_path,
        image_size=image_size,
        opset=opset,
        dynamic_axes=dynamic_axes,
        legacy=legacy,
        external_data=external_data,
        explicit_depth=explicit_depth,
        device=next(model.parameters()).device,
    )
    
    size_mb = sum(p.stat().st_size for p in Path(output_path).parent.glob(f"{Path(output_path).name}*")) / 1e6
    print(f"  ONNX model and external data: {size_mb:.1f} MB")


def export_torchscript(model, output_path, image_size=256):
    print(f"Exporting TorchScript to {output_path}...")
    dummy_content = torch.randn(1, 3, image_size, image_size)
    dummy_style = torch.randn(1, 3, image_size, image_size)

    traced = torch.jit.trace(model, (dummy_content, dummy_style))
    traced.save(output_path)

    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"  TorchScript model: {size_mb:.1f} MB")


def benchmark_latency(model, device, image_size=256, n_runs=50):
    print(f"\nBenchmarking latency ({n_runs} runs, {image_size}x{image_size})...")
    dummy_c = torch.randn(1, 3, image_size, image_size).to(device)
    dummy_s = torch.randn(1, 3, image_size, image_size).to(device)

    model.to(device)
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model(dummy_c, dummy_s)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy_c, dummy_s)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    times = sorted(times)
    print(f"  Mean:   {sum(times)/len(times):.1f} ms")
    print(f"  Median: {times[len(times)//2]:.1f} ms")
    print(f"  P95:    {times[int(len(times)*0.95)]:.1f} ms")
    print(f"  FPS:    {1000 / (sum(times)/len(times)):.1f}")
    return sum(times) / len(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="exports")
    parser.add_argument("--format", type=str, default="all",
                        choices=["onnx", "torchscript", "all"])
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--legacy_onnx", action="store_true")
    parser.add_argument("--static_onnx_shapes", action="store_true")
    parser.add_argument("--no_external_data", action="store_true")
    parser.add_argument("--use_depth", action="store_true")
    parser.add_argument("--depth_scale", type=float, default=2.0)
    parser.add_argument("--benchmark", action="store_true", default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    device = torch.device("cpu")  # Export on CPU
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if args.use_depth:
        full_model = DepthAwareStyleTransferNet(depth_scale=args.depth_scale).to(device)
        full_model.load_rgb_checkpoint(args.checkpoint, device)
    else:
        full_model = StyleTransferNet().to(device)
        full_model.load_state_dict(ckpt["model_state_dict"])
    full_model.eval()

    # Create inference-only model
    model = build_inference_model(
        full_model,
        alpha=args.alpha,
        explicit_depth=args.use_depth,
    )
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    if args.format in ("onnx", "all"):
        try:
            export_onnx(
                model,
                str(output_dir / "style_transfer.onnx"),
                args.image_size,
                opset=args.opset,
                dynamic_axes=not args.static_onnx_shapes,
                legacy=args.legacy_onnx,
                external_data=not args.no_external_data,
                explicit_depth=args.use_depth,
            )
        except Exception as e:
            print(f"  ONNX export failed: {e}")

    if args.format in ("torchscript", "all"):
        try:
            export_torchscript(model, str(output_dir / "style_transfer.pt"), args.image_size)
        except Exception as e:
            print(f"  TorchScript export failed: {e}")

    if args.benchmark:
        gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.use_depth:
            print("Skipping PyTorch benchmark for explicit-depth export wrapper.")
        else:
            benchmark_latency(model, gpu, args.image_size)


if __name__ == "__main__":
    main()
