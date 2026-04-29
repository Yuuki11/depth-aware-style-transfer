#!/usr/bin/env python3
"""Gradio web demo for depth-aware style transfer.

Deploy to HuggingFace Spaces or run locally:
    python app.py --checkpoint checkpoints/best_model.pt

For Spaces, set CHECKPOINT_PATH env var or place model at checkpoints/best_model.pt
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
import gradio as gr

from models import StyleTransferNet, DepthAwareStyleTransferNet
from models.encoder import ResNetEncoder
from utils import get_transform, denormalize

# --- Global model ---
MODEL = None
DEVICE = None


def load_model(checkpoint_path: str, use_depth: bool = False):
    global MODEL, DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_depth:
        MODEL = DepthAwareStyleTransferNet().to(DEVICE)
        MODEL.load_rgb_checkpoint(checkpoint_path, DEVICE)
    else:
        MODEL = StyleTransferNet().to(DEVICE)
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        MODEL.load_state_dict(ckpt["model_state_dict"])

    MODEL.eval()
    print(f"Model loaded on {DEVICE}")


def stylize(
    content_image: Image.Image,
    style_image: Image.Image,
    alpha: float = 1.0,
    output_size: int = 512,
) -> Image.Image:
    """Run style transfer and return PIL image."""
    if MODEL is None:
        return Image.new("RGB", (256, 256), (128, 128, 128))

    transform = get_transform(output_size)
    content_t = transform(content_image).unsqueeze(0).to(DEVICE)
    style_t = transform(style_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = MODEL(content_t, style_t, alpha=alpha)

    output_dn = denormalize(output).squeeze(0).cpu()
    output_np = (output_dn.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(output_np)


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Depth-Aware Style Transfer",
    ) as demo:
        gr.Markdown(
            """
            # Depth-Aware Style Transfer with ResNet34 + AdaIN
            Upload a content image and a style image, adjust the style strength, and see the result.
            
            **Architecture:** ResNet34 encoder → multi-level AdaIN → U-Net decoder with skip connections.
            """
        )

        with gr.Row():
            with gr.Column():
                content_input = gr.Image(label="Content Image", type="pil")
                style_input = gr.Image(label="Style Image", type="pil")
                alpha_slider = gr.Slider(
                    0.0, 1.0, value=1.0, step=0.05,
                    label="Style Strength (α)",
                    info="0 = content only, 1 = full style transfer",
                )
                size_slider = gr.Slider(
                    256, 1024, value=512, step=128,
                    label="Output Resolution",
                )
                run_btn = gr.Button("Stylize", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="Stylized Output", type="pil")

        # Example pairs
        gr.Examples(
            examples=[
                ["assets/sample_content/000000021447.jpg",
                 "assets/sample_style/a.y.-jackson_hills-at-great-bear-lake-1953.jpg",
                 1.0, 512],
            ],
            inputs=[content_input, style_input, alpha_slider, size_slider],
            outputs=output_image,
            fn=stylize,
            cache_examples=False,
        )

        run_btn.click(
            fn=stylize,
            inputs=[content_input, style_input, alpha_slider, size_slider],
            outputs=output_image,
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default=os.environ.get("CHECKPOINT_PATH", "checkpoints/best_model.pt"))
    parser.add_argument("--use_depth", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    load_model(args.checkpoint, args.use_depth)
    demo = build_demo()
    demo.launch(server_port=args.port, share=args.share, theme=gr.themes.Soft())
