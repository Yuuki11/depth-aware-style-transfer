#!/usr/bin/env python3
"""Apply style transfer to video frames.

Usage:
    python scripts/video_style_transfer.py \
        --input video.mp4 --style style.jpg \
        --checkpoint checkpoints/best_model.pt \
        --output stylized_video.mp4

    # With temporal smoothing
    python scripts/video_style_transfer.py \
        --input video.mp4 --style style.jpg \
        --checkpoint checkpoints/best_model.pt \
        --temporal_weight 0.7
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import StyleTransferNet
from utils import get_transform, denormalize


def process_video(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = StyleTransferNet().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load style image
    transform = get_transform(args.size)
    style_img = Image.open(args.style).convert("RGB")
    style_tensor = transform(style_img).unsqueeze(0).to(device)

    # Open video
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Input: {args.input} ({width}x{height}, {fps:.1f}fps, {total_frames} frames)")

    # Output writer
    out_size = (args.size, args.size)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, out_size)

    prev_output = None
    frame_count = 0
    t0 = time.time()

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR -> RGB -> PIL -> tensor
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            content_tensor = transform(pil_frame).unsqueeze(0).to(device)

            # Style transfer
            output = model(content_tensor, style_tensor, alpha=args.alpha)

            # Temporal smoothing (blend with previous frame)
            if prev_output is not None and args.temporal_weight > 0:
                output = (args.temporal_weight * prev_output +
                         (1 - args.temporal_weight) * output)

            prev_output = output.clone()

            # Convert back to numpy
            output_dn = denormalize(output).squeeze(0).cpu()
            output_np = (output_dn.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)

            out.write(output_bgr)
            frame_count += 1

            if frame_count % 30 == 0:
                elapsed = time.time() - t0
                fps_actual = frame_count / elapsed
                print(f"  Frame {frame_count}/{total_frames} "
                      f"({fps_actual:.1f} fps, "
                      f"ETA: {(total_frames-frame_count)/fps_actual:.0f}s)")

    cap.release()
    out.release()

    elapsed = time.time() - t0
    print(f"\nDone: {frame_count} frames in {elapsed:.1f}s "
          f"({frame_count/elapsed:.1f} fps)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--style", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="stylized_video.mp4")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--temporal_weight", type=float, default=0.0,
                        help="Blend with previous frame (0=off, 0.7=smooth)")
    args = parser.parse_args()
    process_video(args)
