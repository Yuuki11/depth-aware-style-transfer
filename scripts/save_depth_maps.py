#!/usr/bin/env python3
"""Save Depth Anything V2 depth maps for image files."""

import argparse
import sys
from pathlib import Path

import torch
import torchvision.utils as vutils

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.depth import DepthEstimator
from utils import load_image


def parse_args():
    parser = argparse.ArgumentParser(description="Generate grayscale depth maps.")
    parser.add_argument("images", nargs="+", help="Input image paths")
    parser.add_argument("--output_dir", default="assets/results/depth_maps")
    parser.add_argument("--image_size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    estimator = DepthEstimator(device=device)

    for image_path in args.images:
        path = Path(image_path)
        image = load_image(path, args.image_size).to(device)
        depth = estimator.from_tensor(image).cpu()
        output_path = output_dir / f"{path.stem}_depth.png"
        vutils.save_image(depth, output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
