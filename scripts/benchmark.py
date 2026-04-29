#!/usr/bin/env python3
"""Benchmark style transfer model — compute metrics for resume/README.

Outputs a formatted table with:
    - SSIM (content preservation)
    - LPIPS (perceptual quality)  
    - Style distance (style transfer quality)
    - Inference latency (ms)
    - Model size (params, MB)
    - FLOPs estimate

Usage:
    python scripts/benchmark.py --checkpoint checkpoints/best_model.pt \
        --content_dir assets/sample_content --style_dir assets/sample_style

    # Full benchmark with VGG comparison
    python scripts/benchmark.py --checkpoint checkpoints/best_model.pt \
        --content_dir data/test_content --style_dir data/test_style \
        --n_pairs 100 --compare_vgg
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import StyleTransferNet, AdaIN
from models.encoder import ResNetEncoder
from utils import load_image, denormalize, get_transform


def compute_ssim(img1, img2, window_size=11):
    C1, C2 = 0.01**2, 0.03**2
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=img1.device).view(1,3,1,1)
    g1 = (img1 * weights).sum(1, keepdim=True)
    g2 = (img2 * weights).sum(1, keepdim=True)
    pad = window_size // 2
    mu1 = nn.functional.avg_pool2d(g1, window_size, 1, pad)
    mu2 = nn.functional.avg_pool2d(g2, window_size, 1, pad)
    s1 = nn.functional.avg_pool2d(g1**2, window_size, 1, pad) - mu1**2
    s2 = nn.functional.avg_pool2d(g2**2, window_size, 1, pad) - mu2**2
    s12 = nn.functional.avg_pool2d(g1*g2, window_size, 1, pad) - mu1*mu2
    ssim = ((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1**2+mu2**2+C1)*(s1+s2+C2))
    return ssim.mean().item()


def compute_style_dist(encoder, output, style):
    adain = AdaIN()
    with torch.no_grad():
        of = encoder(output)
        sf = encoder(style)
    d = 0.0
    for o, s in zip(of, sf):
        om, os_ = adain.calc_mean_std(o)
        sm, ss_ = adain.calc_mean_std(s)
        d += (nn.functional.mse_loss(om, sm) + nn.functional.mse_loss(os_, ss_)).item()
    return d / len(of)


def measure_latency(model, device, size=256, n=50):
    c = torch.randn(1, 3, size, size).to(device)
    s = torch.randn(1, 3, size, size).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(5): model(c, s)
    if device.type == "cuda": torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(n):
            t0 = time.perf_counter()
            model(c, s)
            if device.type == "cuda": torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.median(times), np.percentile(times, 95)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_flops(model, size=256):
    """Rough FLOPs estimate using torch.profiler if available."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        c = torch.randn(1, 3, size, size)
        s = torch.randn(1, 3, size, size)
        model.cpu().eval()
        with FlopCounterMode(model, display=False) as fcm:
            with torch.no_grad():
                model(c, s)
        return fcm.get_total_flops()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--content_dir", type=str, default="assets/sample_content")
    parser.add_argument("--style_dir", type=str, default="assets/sample_style")
    parser.add_argument("--n_pairs", type=int, default=20)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--output", type=str, default="results/benchmark.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = StyleTransferNet().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    encoder = ResNetEncoder(pretrained=False).to(device)
    if "loss_encoder_state_dict" in ckpt:
        encoder.load_state_dict(ckpt["loss_encoder_state_dict"])
    encoder.eval()

    # Model info
    total_p, train_p = count_params(model)
    enc_p, _ = count_params(model.encoder)
    dec_p, _ = count_params(model.decoder)
    
    print("=" * 60)
    print("MODEL INFO")
    print(f"  Encoder params:  {enc_p:>12,} ({enc_p/1e6:.1f}M)")
    print(f"  Decoder params:  {dec_p:>12,} ({dec_p/1e6:.1f}M)")
    print(f"  Total params:    {total_p:>12,} ({total_p/1e6:.1f}M)")

    # Latency
    mean_ms, median_ms, p95_ms = measure_latency(model, device, args.size)
    print(f"\nLATENCY ({args.size}x{args.size})")
    print(f"  Mean:   {mean_ms:.1f} ms ({1000/mean_ms:.0f} FPS)")
    print(f"  Median: {median_ms:.1f} ms")
    print(f"  P95:    {p95_ms:.1f} ms")

    # Quality metrics
    content_paths = sorted(Path(args.content_dir).glob("*"))[:args.n_pairs]
    style_paths = sorted(Path(args.style_dir).glob("*"))[:args.n_pairs]

    if content_paths and style_paths:
        ssim_scores, style_dists = [], []

        for i in range(min(len(content_paths), len(style_paths))):
            content = load_image(content_paths[i], args.size).to(device)
            style = load_image(style_paths[i % len(style_paths)], args.size).to(device)

            with torch.no_grad():
                output = model(content, style, alpha=1.0)

            ssim_scores.append(compute_ssim(denormalize(output), denormalize(content)))
            style_dists.append(compute_style_dist(encoder, output, style))

        print(f"\nQUALITY METRICS ({len(ssim_scores)} pairs)")
        print(f"  SSIM (content):     {np.mean(ssim_scores):.4f} ± {np.std(ssim_scores):.4f}")
        print(f"  Style distance:     {np.mean(style_dists):.4f} ± {np.std(style_dists):.4f}")

        # LPIPS if available
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="alex").to(device)
            lpips_scores = []
            for i in range(min(len(content_paths), len(style_paths))):
                content = load_image(content_paths[i], args.size).to(device)
                style = load_image(style_paths[i % len(style_paths)], args.size).to(device)
                with torch.no_grad():
                    output = model(content, style, alpha=1.0)
                    lpips_scores.append(lpips_fn(denormalize(output), denormalize(content)).item())
            print(f"  LPIPS (perceptual): {np.mean(lpips_scores):.4f} ± {np.std(lpips_scores):.4f}")
        except ImportError:
            print("  LPIPS: install with `pip install lpips`")

    # Save results
    results = {
        "model": {
            "encoder": "ResNet34",
            "total_params": total_p,
            "encoder_params": enc_p,
            "decoder_params": dec_p,
        },
        "latency": {
            "mean_ms": round(mean_ms, 1),
            "median_ms": round(median_ms, 1),
            "p95_ms": round(p95_ms, 1),
            "fps": round(1000 / mean_ms, 1),
            "resolution": args.size,
            "device": str(device),
        },
    }
    if content_paths and style_paths:
        results["quality"] = {
            "ssim": round(np.mean(ssim_scores), 4),
            "style_distance": round(np.mean(style_dists), 4),
            "n_pairs": len(ssim_scores),
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
