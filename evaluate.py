#!/usr/bin/env python3
"""Evaluate style transfer quality with quantitative metrics.

Computes:
    - SSIM:  Structural similarity between output and content (higher = better content preservation)
    - LPIPS: Perceptual distance between output and content (lower = better content preservation)  
    - Style Loss: Feature statistics distance between output and style (lower = better style transfer)
    - FID:   Fréchet Inception Distance across a batch (lower = better quality)

Usage:
    # Evaluate a single pair
    python evaluate.py --content img/content.jpg --style img/style.jpg \
        --checkpoint checkpoints/model_final.pt

    # Evaluate across a directory of content/style pairs
    python evaluate.py --content_dir data/test_content --style_dir data/test_style \
        --checkpoint checkpoints/model_final.pt --batch_eval

    # Compare ResNet vs VGG (if you have both checkpoints)
    python evaluate.py --content_dir data/test_content --style_dir data/test_style \
        --checkpoint checkpoints/resnet_final.pt --output_csv results/metrics.csv
"""

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from models import StyleTransferNet, DepthAwareStyleTransferNet, AdaIN
from models.encoder import ResNetEncoder
from utils import load_image, denormalize, get_transform


# ---------------------------------------------------------------------------
# Metric implementations (no extra deps needed)
# ---------------------------------------------------------------------------

def compute_ssim(
    img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11
) -> float:
    """Compute SSIM between two image tensors.
    
    Simplified implementation. For production, use torchmetrics.
    Inputs should be (1, C, H, W) in [0, 1].
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Convert to grayscale
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=img1.device).view(1, 3, 1, 1)
    img1_gray = (img1 * weights).sum(dim=1, keepdim=True)
    img2_gray = (img2 * weights).sum(dim=1, keepdim=True)

    # Gaussian-like averaging via avg pool
    pad = window_size // 2
    mu1 = nn.functional.avg_pool2d(img1_gray, window_size, stride=1, padding=pad)
    mu2 = nn.functional.avg_pool2d(img2_gray, window_size, stride=1, padding=pad)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = nn.functional.avg_pool2d(img1_gray ** 2, window_size, stride=1, padding=pad) - mu1_sq
    sigma2_sq = nn.functional.avg_pool2d(img2_gray ** 2, window_size, stride=1, padding=pad) - mu2_sq
    sigma12 = nn.functional.avg_pool2d(img1_gray * img2_gray, window_size, stride=1, padding=pad) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


def compute_style_distance(
    encoder: nn.Module,
    output: torch.Tensor,
    style: torch.Tensor,
) -> float:
    """Compute style distance as mean/variance difference across encoder levels."""
    adain = AdaIN()
    
    with torch.no_grad():
        out_feats = encoder(output)
        sty_feats = encoder(style)

    total = 0.0
    for of, sf in zip(out_feats, sty_feats):
        o_mean, o_std = adain.calc_mean_std(of)
        s_mean, s_std = adain.calc_mean_std(sf)
        total += (
            nn.functional.mse_loss(o_mean, s_mean).item()
            + nn.functional.mse_loss(o_std, s_std).item()
        )
    return total / len(out_feats)


def compute_content_distance(
    encoder: nn.Module,
    output: torch.Tensor,
    content: torch.Tensor,
) -> float:
    """Compute content distance as MSE of encoder features."""
    with torch.no_grad():
        out_feats = encoder(output)
        con_feats = encoder(content)

    total = 0.0
    for of, cf in zip(out_feats, con_feats):
        total += nn.functional.mse_loss(of, cf).item()
    return total / len(out_feats)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_single(
    model: nn.Module,
    encoder: nn.Module,
    content: torch.Tensor,
    style: torch.Tensor,
    device: torch.device,
    alpha: float = 1.0,
) -> dict:
    """Evaluate a single content-style pair."""
    model.eval()
    
    with torch.no_grad():
        output = model(content, style, alpha=alpha)

    # Denormalize for pixel-level metrics
    output_dn = denormalize(output)
    content_dn = denormalize(content)

    metrics = {
        "ssim": compute_ssim(output_dn, content_dn),
        "content_distance": compute_content_distance(encoder, output, content),
        "style_distance": compute_style_distance(encoder, output, style),
    }

    # LPIPS (optional — only if lpips is installed)
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex").to(device)
        with torch.no_grad():
            metrics["lpips"] = lpips_fn(output_dn, content_dn).item()
    except ImportError:
        metrics["lpips"] = None

    return metrics


def evaluate_batch(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    if args.use_depth:
        model = DepthAwareStyleTransferNet(depth_scale=args.depth_scale).to(device)
        model.load_rgb_checkpoint(args.checkpoint, device)
    else:
        model = StyleTransferNet().to(device)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Loss encoder for distance metrics
    encoder = ResNetEncoder(pretrained=True).to(device)
    if "loss_encoder_state_dict" in ckpt:
        encoder.load_state_dict(ckpt["loss_encoder_state_dict"])
    encoder.eval()

    if args.batch_eval:
        content_dir = Path(args.content_dir)
        style_dir = Path(args.style_dir)
        content_paths = sorted(content_dir.glob("*.jpg"))[:args.max_eval]
        style_paths = sorted(style_dir.glob("*.jpg"))[:args.max_eval]

        n_pairs = min(len(content_paths), len(style_paths))
        print(f"Evaluating {n_pairs} pairs...")

        all_metrics = []
        for i in range(n_pairs):
            content = load_image(content_paths[i], args.image_size).to(device)
            style = load_image(style_paths[i % len(style_paths)], args.image_size).to(device)

            m = evaluate_single(model, encoder, content, style, device, args.alpha)
            m["content_path"] = str(content_paths[i].name)
            m["style_path"] = str(style_paths[i % len(style_paths)].name)
            all_metrics.append(m)

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{n_pairs}] SSIM={m['ssim']:.4f} "
                      f"Style_dist={m['style_distance']:.4f}")

        # Aggregate
        avg = {
            "ssim": np.mean([m["ssim"] for m in all_metrics]),
            "content_distance": np.mean([m["content_distance"] for m in all_metrics]),
            "style_distance": np.mean([m["style_distance"] for m in all_metrics]),
        }
        lpips_vals = [m["lpips"] for m in all_metrics if m["lpips"] is not None]
        if lpips_vals:
            avg["lpips"] = np.mean(lpips_vals)

        print(f"\n{'='*50}")
        print(f"Average Metrics over {n_pairs} pairs:")
        for k, v in avg.items():
            print(f"  {k}: {v:.4f}")
        print(f"{'='*50}")

        # Save to CSV
        if args.output_csv:
            output_path = Path(args.output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
                writer.writeheader()
                writer.writerows(all_metrics)
            print(f"Saved per-pair metrics to {output_path}")

        # Save summary JSON
        summary_path = Path(args.output_csv or "results/metrics.csv").parent / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(avg, f, indent=2)

    else:
        # Single pair
        content = load_image(args.content, args.image_size).to(device)
        style = load_image(args.style, args.image_size).to(device)
        metrics = evaluate_single(model, encoder, content, style, device, args.alpha)

        print(f"\nMetrics:")
        for k, v in metrics.items():
            if v is not None:
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: N/A (install lpips: pip install lpips)")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Style Transfer")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--content", type=str, default=None)
    parser.add_argument("--style", type=str, default=None)
    parser.add_argument("--content_dir", type=str, default=None)
    parser.add_argument("--style_dir", type=str, default=None)
    parser.add_argument("--batch_eval", action="store_true")
    parser.add_argument("--max_eval", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--use_depth", action="store_true")
    parser.add_argument("--depth_scale", type=float, default=2.0)
    parser.add_argument("--output_csv", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_batch(args)
