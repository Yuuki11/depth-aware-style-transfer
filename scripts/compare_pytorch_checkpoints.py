#!/usr/bin/env python3
"""Compare PyTorch outputs from multiple style-transfer checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import compute_ssim
from models import DepthAwareStyleTransferNet, StyleTransferNet
from quantization.modelopt_utils import prepare_modelopt_qat
from utils import denormalize, load_image, save_image


def parse_epoch(path: Path) -> int:
    match = re.search(r"epoch[_-](\d+)", path.stem)
    return int(match.group(1)) if match else -1


def psnr(mse: float) -> float:
    return float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = denormalize(tensor.detach().cpu()).squeeze(0).permute(1, 2, 0).numpy()
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def build_model(
    checkpoint_path: Path,
    device: torch.device,
    *,
    use_depth: bool,
    depth_scale: float,
    content: torch.Tensor,
    style: torch.Tensor,
):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    qat_cfg = ckpt.get("qat")

    if use_depth:
        model = DepthAwareStyleTransferNet(depth_scale=depth_scale).to(device)
    else:
        model = StyleTransferNet().to(device)

    if isinstance(qat_cfg, dict) and qat_cfg.get("enabled"):
        def forward_loop(module: torch.nn.Module) -> None:
            module(content, style)

        model = prepare_modelopt_qat(
            model,
            forward_loop,
            mode=qat_cfg.get("mode", "int8"),
            algorithm=qat_cfg.get("algorithm", "max"),
            disable_encoder=bool(qat_cfg.get("disable_encoder", False)),
            disable_decoder=bool(qat_cfg.get("disable_decoder", False)),
            disable_final_layer=bool(qat_cfg.get("disable_final_layer", False)),
            extra_disable_patterns=[
                p.strip()
                for p in str(qat_cfg.get("extra_disable_patterns", "")).split(",")
                if p.strip()
            ],
        )

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if unexpected:
        print(f"{checkpoint_path.name}: unexpected keys ignored: {unexpected}")
    if missing:
        print(f"{checkpoint_path.name}: missing keys: {missing}")

    model.eval()
    return model, ckpt


def make_grid(items: list[tuple[str, Image.Image]], output_path: Path, columns: int) -> None:
    width, height = items[0][1].size
    label_h = 30
    rows = math.ceil(len(items) / columns)
    canvas = Image.new("RGB", (columns * width, rows * (height + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, img) in enumerate(items):
        row = idx // columns
        col = idx % columns
        x = col * width
        y = row * (height + label_h)
        canvas.paste(img, (x, y + label_h))
        draw.text((x + 8, y + 8), label, fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PyTorch checkpoint outputs")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="checkpoint_epoch_*.pt")
    parser.add_argument("--content", type=str, required=True)
    parser.add_argument("--style", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/checkpoint_comparison")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--use_depth", action="store_true")
    parser.add_argument("--depth_scale", type=float, default=2.0)
    parser.add_argument("--columns", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob(args.pattern), key=parse_epoch)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched {checkpoint_dir / args.pattern}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoints: {len(checkpoints)}")

    content = load_image(args.content, args.image_size).to(device)
    style = load_image(args.style, args.image_size).to(device)
    content_dn = denormalize(content)

    outputs: list[tuple[int, Path, torch.Tensor, dict[str, float | int | str | None]]] = []
    for checkpoint_path in checkpoints:
        epoch = parse_epoch(checkpoint_path)
        print(f"Running {checkpoint_path.name}...")
        model, ckpt = build_model(
            checkpoint_path,
            device,
            use_depth=args.use_depth,
            depth_scale=args.depth_scale,
            content=content,
            style=style,
        )
        with torch.no_grad():
            output = model(content, style, alpha=args.alpha)

        output_path = output_dir / f"epoch{epoch:03d}_output.png"
        save_image(output, output_path)

        output_dn = denormalize(output)
        content_mse = torch.mean((output_dn - content_dn) ** 2).item()
        content_mae = torch.mean(torch.abs(output_dn - content_dn)).item()
        row: dict[str, float | int | str | None] = {
            "epoch": epoch,
            "checkpoint": str(checkpoint_path),
            "output": str(output_path),
            "training_content_loss": ckpt.get("content_loss"),
            "training_style_loss": ckpt.get("style_loss"),
            "ssim_vs_content": compute_ssim(output_dn, content_dn),
            "mae_vs_content": content_mae,
            "mse_vs_content": content_mse,
            "psnr_vs_content_db": psnr(content_mse),
        }
        outputs.append((epoch, output_path, output.detach().cpu(), row))

    final_epoch, _, final_output, _ = outputs[-1]
    previous_output = None
    rows = []
    for epoch, _, output, row in outputs:
        final_mse = torch.mean((denormalize(output) - denormalize(final_output)) ** 2).item()
        row["mse_vs_final_epoch"] = final_mse
        row["psnr_vs_final_epoch_db"] = psnr(final_mse)
        if previous_output is None:
            row["mae_vs_previous_epoch"] = None
        else:
            row["mae_vs_previous_epoch"] = torch.mean(
                torch.abs(denormalize(output) - denormalize(previous_output))
            ).item()
        previous_output = output
        rows.append(row)

    metrics_json = output_dir / "metrics.json"
    metrics_csv = output_dir / "metrics.csv"
    metrics_json.write_text(json.dumps(rows, indent=2))
    with metrics_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    grid_items = [
        ("Content", Image.open(args.content).convert("RGB").resize((args.image_size, args.image_size))),
        ("Style", Image.open(args.style).convert("RGB").resize((args.image_size, args.image_size))),
    ]
    for epoch, output_path, _, _ in outputs:
        grid_items.append((f"Epoch {epoch}", Image.open(output_path).convert("RGB")))

    grid_path = output_dir / "epoch_comparison_grid.png"
    make_grid(grid_items, grid_path, columns=args.columns)

    print(f"Saved grid: {grid_path}")
    print(f"Saved metrics: {metrics_json}")
    print(f"Final comparison reference epoch: {final_epoch}")


if __name__ == "__main__":
    main()
