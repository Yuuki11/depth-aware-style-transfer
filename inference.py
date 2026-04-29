#!/usr/bin/env python3
"""Run style transfer inference.

Usage:
    # Basic style transfer
    python inference.py --content img/content.jpg --style img/style.jpg \
        --checkpoint checkpoints/model_final.pt --output output.png

    # Depth-aware style transfer
    python inference.py --content img/content.jpg --style img/style.jpg \
        --checkpoint checkpoints/model_final.pt --output output.png \
        --use_depth --depth_scale 0.1

    # Adjust style strength
    python inference.py --content img/content.jpg --style img/style.jpg \
        --checkpoint checkpoints/model_final.pt --alpha 0.5
"""

import argparse
from pathlib import Path

import torch

from models import StyleTransferNet, DepthAwareStyleTransferNet
from utils import load_image, save_image, save_comparison_grid


def run_inference(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load images
    content = load_image(args.content, size=args.image_size).to(device)
    style = load_image(args.style, size=args.image_size).to(device)

    # Load model
    if args.use_depth:
        model = DepthAwareStyleTransferNet(depth_scale=args.depth_scale).to(device)
        model.load_rgb_checkpoint(args.checkpoint, device)
    else:
        model = StyleTransferNet().to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()

    # Style transfer
    with torch.no_grad():
        output = model(content, style, alpha=args.alpha)

    # Save
    output_path = Path(args.output)
    save_image(output, output_path)
    print(f"Saved: {output_path}")

    # Save comparison grid
    if args.save_grid:
        grid_path = output_path.parent / f"{output_path.stem}_comparison.png"
        save_comparison_grid(content, style, output, grid_path)
        print(f"Saved comparison: {grid_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Style Transfer Inference")
    parser.add_argument("--content", type=str, required=True, help="Content image path")
    parser.add_argument("--style", type=str, required=True, help="Style image path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--output", type=str, default="output.png", help="Output image path")
    parser.add_argument("--alpha", type=float, default=1.0, help="Style strength (0-1)")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--use_depth", action="store_true")
    parser.add_argument("--depth_scale", type=float, default=0.1)
    parser.add_argument("--save_grid", action="store_true", help="Save side-by-side comparison")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
