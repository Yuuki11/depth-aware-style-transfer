#!/usr/bin/env python3
"""Generate comparison visualizations and loss plots.

Usage:
    # Plot training loss curves
    python visualize.py --plot_loss --history checkpoints/training_history.json

    # Generate comparison grid from a directory of content/style images
    python visualize.py --compare --checkpoint checkpoints/model_final.pt \
        --content_dir assets/sample_content --style_dir assets/sample_style

    # Alpha sweep (show effect of style strength)
    python visualize.py --alpha_sweep --checkpoint checkpoints/model_final.pt \
        --content assets/sample_content/000000021447.jpg \
        --style assets/sample_style/a.y.-jackson_hills-at-great-bear-lake-1953.jpg
"""

import argparse
import json
from pathlib import Path

import torch
import torchvision.utils as vutils
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import StyleTransferNet
from utils import load_image, denormalize


def plot_training_loss(history_path: str, output_path: str = "assets/results/loss_curves.png") -> None:
    """Plot content/style/total loss curves from training history."""
    with open(history_path) as f:
        history = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["content_loss"]) + 1)

    ax1.plot(epochs, history["content_loss"], "b-o", label="Content Loss", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Content Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["style_loss"], "r-o", label="Style Loss", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Style Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved loss curves to {output_path}")


def generate_comparison_grid(
    checkpoint_path: str,
    content_dir: str,
    style_dir: str,
    output_path: str = "assets/results/comparison_grid.png",
    n_content: int = 4,
    n_style: int = 4,
    image_size: int = 256,
) -> None:
    """Generate an NxM grid: rows = content images, columns = style images."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StyleTransferNet().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    content_paths = sorted(Path(content_dir).glob("*.jpg"))[:n_content]
    style_paths = sorted(Path(style_dir).glob("*.jpg"))[:n_style]

    # First row: empty + style images
    # Subsequent rows: content image + stylized outputs
    rows = []

    # Header row: placeholder + style images
    header = [torch.ones(1, 3, image_size, image_size)]  # blank
    for sp in style_paths:
        header.append(denormalize(load_image(sp, image_size)))
    rows.append(torch.cat(header, dim=0))

    # Content rows
    for cp in content_paths:
        content = load_image(cp, image_size).to(device)
        row = [denormalize(content.cpu())]

        for sp in style_paths:
            style = load_image(sp, image_size).to(device)
            with torch.no_grad():
                output = model(content, style, alpha=1.0)
            row.append(denormalize(output.cpu()))

        rows.append(torch.cat(row, dim=0))

    # Combine all rows
    all_images = torch.cat(rows, dim=0)
    ncol = len(style_paths) + 1

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(all_images, output_path, nrow=ncol, padding=4, pad_value=1)
    print(f"Saved comparison grid to {output_path}")


def alpha_sweep(
    checkpoint_path: str,
    content_path: str,
    style_path: str,
    output_path: str = "assets/results/alpha_sweep.png",
    image_size: int = 256,
) -> None:
    """Show style transfer at different alpha values (0.0 to 1.0)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StyleTransferNet().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    content = load_image(content_path, image_size).to(device)
    style = load_image(style_path, image_size).to(device)

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    images = [denormalize(content.cpu()), denormalize(style.cpu())]

    for a in alphas:
        with torch.no_grad():
            output = model(content, style, alpha=a)
        images.append(denormalize(output.cpu()))

    all_images = torch.cat(images, dim=0)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(all_images, output_path, nrow=len(images), padding=4)
    print(f"Saved alpha sweep to {output_path}")
    print(f"  Order: Content | Style | alpha={' | alpha='.join(str(a) for a in alphas)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Style Transfer Results")
    parser.add_argument("--plot_loss", action="store_true")
    parser.add_argument("--history", type=str, default="checkpoints/training_history.json")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--alpha_sweep", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--content", type=str, default=None)
    parser.add_argument("--style", type=str, default=None)
    parser.add_argument("--content_dir", type=str, default="assets/sample_content")
    parser.add_argument("--style_dir", type=str, default="assets/sample_style")
    parser.add_argument("--image_size", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.plot_loss:
        plot_training_loss(args.history)

    if args.compare:
        generate_comparison_grid(args.checkpoint, args.content_dir, args.style_dir)

    if args.alpha_sweep:
        alpha_sweep(args.checkpoint, args.content, args.style)
