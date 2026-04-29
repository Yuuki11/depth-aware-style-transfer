#!/usr/bin/env python3
"""Create a side-by-side comparison grid for depth-aware checkpoints."""

import argparse
import sys
from pathlib import Path

import torch
import torchvision.utils as vutils
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import DepthAwareStyleTransferNet
from utils import denormalize, load_image


def parse_args():
    parser = argparse.ArgumentParser(description="Compare depth-aware checkpoints.")
    parser.add_argument("--content", required=True)
    parser.add_argument("--style", required=True)
    parser.add_argument("--checkpoint_dir", default="checkpoints/resnet_adain_depth")
    parser.add_argument("--epochs", default="1,2,3,4")
    parser.add_argument("--output", default="assets/results/depth_epoch_comparison.png")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--depth_scale", type=float, default=2.0)
    return parser.parse_args()


def add_labels(image_path: Path, labels: list[str], tile_size: int) -> None:
    image = Image.open(image_path).convert("RGB")
    label_height = 28
    labeled = Image.new("RGB", (image.width, image.height + label_height), "white")
    labeled.paste(image, (0, label_height))
    draw = ImageDraw.Draw(labeled)
    for idx, label in enumerate(labels):
        x = idx * tile_size + 8
        draw.text((x, 7), label, fill=(0, 0, 0))
    labeled.save(image_path)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = [int(e.strip()) for e in args.epochs.split(",") if e.strip()]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = load_image(args.content, args.image_size).to(device)
    style = load_image(args.style, args.image_size).to(device)
    model = DepthAwareStyleTransferNet(depth_scale=args.depth_scale).to(device).eval()

    images = [denormalize(content.cpu()), denormalize(style.cpu())]
    labels = ["Content", "Style"]

    for epoch in epochs:
        checkpoint = Path(args.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
        model.load_rgb_checkpoint(str(checkpoint), device)
        with torch.no_grad():
            output = model(content, style, alpha=1.0)
        images.append(denormalize(output.cpu()))
        labels.append(f"Epoch {epoch}")

    grid = torch.cat(images, dim=0)
    vutils.save_image(grid, output_path, nrow=len(images), padding=4, pad_value=1)
    add_labels(output_path, labels, args.image_size + 4)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
