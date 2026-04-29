#!/usr/bin/env python3
"""Train ResNet34-AdaIN style transfer model.

Usage:
    # Default: train on COCO + WikiArt
    python train.py --content_dir ./data/coco/train2017 --style_dir ./data/wikiart

    # Quick test with fewer images
    python train.py --content_dir ./data/coco/train2017 --style_dir ./data/wikiart \
        --max_content_images 1000 --max_style_images 1000 --epochs 5

    # Single style (Art Nouveau Modern)
    python train.py --style_dir ./data/wikiart/Art_Nouveau_Modern --epochs 50

    # Single style with the improved frozen-ResNet AdaIN objective
    python train.py --content_dir ./data/coco/train2017 \
        --style_dir ./data/wikiart/Art_Nouveau_Modern \
        --checkpoint_dir ./checkpoints/resnet_adain_single \
        --sample_dir ./samples/resnet_adain_single \
        --epochs 25 --batch_size 8 --amp

    # Depth-aware training with the same stable objective
    python train.py --use_depth --content_dir ./data/coco/train2017 \
        --style_dir ./data/wikiart/Art_Nouveau_Modern \
        --checkpoint_dir ./checkpoints/resnet_adain_depth \
        --sample_dir ./samples/resnet_adain_depth \
        --epochs 25 --batch_size 4 --depth_scale 2.0 --amp
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import StyleTransferNet, DepthAwareStyleTransferNet, AdaIN
from models.encoder import ResNetEncoder
from quantization.calibration import run_calibration_forward_loop
from quantization.modelopt_utils import prepare_modelopt_qat
from utils import get_train_transform, save_image
from utils.dataset import ImageFolderDataset


DEFAULT_CONTENT_FEATURE_WEIGHTS = [0.0, 0.0, 0.0, 0.5, 0.5]
DEFAULT_STYLE_FEATURE_WEIGHTS = [1.0, 1.0, 1.0, 0.5, 0.25]


def parse_feature_weights(raw: str | None, defaults: list[float]) -> list[float]:
    """Parse comma-separated per-level feature weights."""
    if raw is None:
        weights = defaults
    else:
        weights = [float(x.strip()) for x in raw.split(",") if x.strip()]
        if len(weights) != len(defaults):
            raise ValueError(f"Expected {len(defaults)} feature weights, got {len(weights)}")

    total = sum(weights)
    if total <= 0:
        raise ValueError("At least one feature weight must be positive")
    return [w / total for w in weights]


def freeze_module(module: nn.Module) -> None:
    """Freeze parameters and keep BatchNorm/Dropout in inference mode."""
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def compute_content_loss_from_features(
    generated_features: list[torch.Tensor],
    target_features: list[torch.Tensor],
    weights: list[float] | None = None,
) -> torch.Tensor:
    """Content loss against the AdaIN target features."""
    if weights is None:
        weights = [1.0 / len(generated_features)] * len(generated_features)

    loss = torch.tensor(0.0, device=generated_features[0].device)
    for w, generated, target in zip(weights, generated_features, target_features):
        if w > 0:
            loss += w * nn.functional.mse_loss(generated, target.detach())
    return loss


def compute_style_loss_from_features(
    generated_features: list[torch.Tensor],
    style_features: list[torch.Tensor],
    weights: list[float] | None = None,
) -> torch.Tensor:
    """Multi-level style loss using mean/std feature statistics."""
    adain = AdaIN()

    if weights is None:
        weights = [1.0 / len(generated_features)] * len(generated_features)

    loss = torch.tensor(0.0, device=generated_features[0].device)
    for w, gf, sf in zip(weights, generated_features, style_features):
        if w <= 0:
            continue
        g_mean, g_std = adain.calc_mean_std(gf)
        s_mean, s_std = adain.calc_mean_std(sf)
        loss += w * (
            nn.functional.mse_loss(g_mean, s_mean.detach())
            + nn.functional.mse_loss(g_std, s_std.detach())
        )
    return loss


def build_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    """Build optimizer with differential learning rates."""
    encoder = model.encoder
    decoder = model.decoder

    param_groups = [{"params": list(decoder.parameters()), "lr": args.lr_decoder}]

    if args.train_encoder:
        param_groups.extend([
            {
                "params": list(encoder.conv1.parameters())
                + list(encoder.bn1.parameters())
                + list(encoder.layer1.parameters())
                + list(encoder.layer2.parameters()),
                "lr": args.lr_encoder_low,
            },
            {
                "params": list(encoder.layer3.parameters())
                + list(encoder.layer4.parameters()),
                "lr": args.lr_encoder_high,
            },
        ])

    return torch.optim.Adam(param_groups, weight_decay=args.weight_decay)


def generate_samples(
    model: nn.Module,
    content_loader: DataLoader,
    style_loader: DataLoader,
    device: torch.device,
    epoch: int,
    sample_dir: Path,
    n_samples: int = 4,
    train_encoder: bool = False,
) -> None:
    """Generate and save sample style transfers."""
    model.eval()
    sample_dir.mkdir(parents=True, exist_ok=True)

    content_iter = iter(content_loader)
    style_iter = iter(style_loader)

    with torch.no_grad():
        for i in range(min(n_samples, len(content_loader), len(style_loader))):
            content = next(content_iter)[:1].to(device)
            style = next(style_iter)[:1].to(device)
            output = model(content, style, alpha=1.0)

            save_image(content, sample_dir / f"epoch{epoch:03d}_sample{i}_content.png")
            save_image(style, sample_dir / f"epoch{epoch:03d}_sample{i}_style.png")
            save_image(output, sample_dir / f"epoch{epoch:03d}_sample{i}_output.png")

    model.train()
    if not train_encoder:
        model.encoder.eval()


def encode_for_generator(
    model: nn.Module,
    images: torch.Tensor,
    train_encoder: bool,
) -> list[torch.Tensor]:
    """Encode images for the generator, optionally freezing the encoder graph."""
    if train_encoder:
        return model.encode(images)
    with torch.no_grad():
        return model.encode(images)


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    avg_c: float,
    avg_s: float,
    args,
    loss_encoder: nn.Module | None = None,
) -> dict:
    """Build a compact checkpoint payload."""
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "content_loss": avg_c,
        "style_loss": avg_s,
        "config": vars(args),
    }
    if getattr(args, "qat", False):
        payload["qat"] = {
            "enabled": True,
            "mode": args.qat_mode,
            "algorithm": args.qat_algorithm,
            "calib_batches": args.qat_calib_batches,
            "disable_encoder": args.qat_disable_encoder,
            "disable_decoder": args.qat_disable_decoder,
            "disable_final_layer": args.qat_disable_final_layer,
            "extra_disable_patterns": args.qat_disable_patterns,
        }
    if args.save_loss_encoder and loss_encoder is not None:
        payload["loss_encoder_state_dict"] = loss_encoder.state_dict()
    return payload


def _checkpoint_has_qat(checkpoint: dict) -> bool:
    qat_meta = checkpoint.get("qat")
    if isinstance(qat_meta, dict):
        return bool(qat_meta.get("enabled", False))
    config = checkpoint.get("config", {})
    return bool(config.get("qat", False)) if isinstance(config, dict) else False


def maybe_prepare_qat_model(
    model: nn.Module,
    content_loader: DataLoader,
    style_loader: DataLoader,
    device: torch.device,
    args,
) -> nn.Module:
    """Insert calibrated ModelOpt fake quantizers when --qat is requested."""
    if not args.qat:
        return model

    if args.use_depth:
        print(
            "Depth-aware QAT is enabled. Training still computes depth with the PyTorch "
            "DepthEstimator, but TensorRT export uses explicit depth-map inputs rather "
            "than exporting Depth Anything inside the engine."
        )

    if args.qat_mode != "int8":
        print(
            f"Warning: --qat_mode {args.qat_mode} is experimental for this Conv/AdaIN model. "
            "INT8 is the recommended QAT mode for TensorRT."
        )

    if not args.resume_from:
        print(
            "Warning: --qat is usually best as short fine-tuning from an FP32 checkpoint. "
            "You are starting QAT from the initial model weights."
        )

    if args.qat_lr_scale != 1.0:
        args.lr_decoder *= args.qat_lr_scale
        args.lr_encoder_low *= args.qat_lr_scale
        args.lr_encoder_high *= args.qat_lr_scale

    patterns = [p.strip() for p in args.qat_disable_patterns.split(",") if p.strip()]

    print(
        f"Preparing ModelOpt QAT ({args.qat_mode}, algorithm={args.qat_algorithm}, "
        f"calib_batches={args.qat_calib_batches})..."
    )

    def forward_loop(qmodel: nn.Module) -> None:
        run_calibration_forward_loop(
            qmodel,
            content_loader,
            style_loader,
            device=device,
            max_batches=args.qat_calib_batches,
        )

    model = prepare_modelopt_qat(
        model,
        forward_loop,
        mode=args.qat_mode,
        algorithm=args.qat_algorithm,
        disable_encoder=args.qat_disable_encoder,
        disable_decoder=args.qat_disable_decoder,
        disable_final_layer=args.qat_disable_final_layer,
        extra_disable_patterns=patterns,
        print_summary=args.qat_print_summary,
    )
    print("ModelOpt QAT is enabled.")
    return model


def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Directories
    ckpt_dir = Path(args.checkpoint_dir)
    sample_dir = Path(args.sample_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    train_tf = get_train_transform(args.image_size)
    content_dataset = ImageFolderDataset(
        args.content_dir, transform=train_tf, max_images=args.max_content_images
    )
    style_dataset = ImageFolderDataset(
        args.style_dir, transform=train_tf, max_images=args.max_style_images
    )
    print(f"Content images: {len(content_dataset)}")
    print(f"Style images:   {len(style_dataset)}")

    content_loader = DataLoader(
        content_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    style_loader = DataLoader(
        style_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if len(content_loader) == 0 or len(style_loader) == 0:
        raise RuntimeError(
            "No training batches were created. Lower --batch_size or add more images."
        )

    content_feature_weights = parse_feature_weights(
        args.content_feature_weights, DEFAULT_CONTENT_FEATURE_WEIGHTS
    )
    style_feature_weights = parse_feature_weights(
        args.style_feature_weights, DEFAULT_STYLE_FEATURE_WEIGHTS
    )

    # Model
    if args.use_depth:
        model = DepthAwareStyleTransferNet(
            depth_scale=args.depth_scale,
            pretrained_encoder=args.pretrained_encoder,
        ).to(device)
    else:
        model = StyleTransferNet(
            pretrained_encoder=args.pretrained_encoder
        ).to(device)

    loss_encoder = ResNetEncoder(pretrained=True).to(device)
    freeze_module(loss_encoder)
    if not args.train_encoder:
        freeze_module(model.encoder)

    start_epoch = 0
    resume_ckpt = None

    # Resume from checkpoint
    if args.resume_from:
        resume_ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        resume_is_qat = _checkpoint_has_qat(resume_ckpt)
        if not args.qat and resume_is_qat:
            raise RuntimeError(
                "This checkpoint was saved from a QAT model. Resume with --qat so the "
                "ModelOpt quantizer modules are recreated before loading."
            )
        if args.qat and resume_is_qat:
            model = maybe_prepare_qat_model(model, content_loader, style_loader, device, args)
            model.load_state_dict(resume_ckpt["model_state_dict"])
        else:
            model.load_state_dict(resume_ckpt["model_state_dict"])

    if args.qat and (resume_ckpt is None or not _checkpoint_has_qat(resume_ckpt)):
        model = maybe_prepare_qat_model(model, content_loader, style_loader, device, args)

    optimizer = build_optimizer(model, args)
    if resume_ckpt is not None and "optimizer_state_dict" in resume_ckpt:
        try:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        except ValueError as exc:
            if args.qat and not _checkpoint_has_qat(resume_ckpt):
                print(f"Skipping FP32 optimizer state after QAT preparation: {exc}")
            else:
                raise
        start_epoch = resume_ckpt.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch}")

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Scheduler
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.T_max
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=20, gamma=0.5
        )

    # Training history
    history = {"content_loss": [], "style_loss": [], "total_loss": []}

    print(f"\n{'='*60}")
    print(f"Training config:")
    print(f"  Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"  Content weight: {args.content_weight}, Style weight: {args.style_weight}")
    print(f"  Pretrained encoder: {args.pretrained_encoder}, Train encoder: {args.train_encoder}")
    print(f"  Depth aware: {args.use_depth}")
    print(f"  AMP: {use_amp}")
    print(f"  QAT: {args.qat}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_encoder.eval()
        if not args.train_encoder:
            model.encoder.eval()

        epoch_content_loss = 0.0
        epoch_style_loss = 0.0
        epoch_total_loss = 0.0
        n_batches = 0
        optimizer_steps = 0

        style_iter = iter(style_loader)
        t0 = time.time()

        for batch_idx, content_batch in enumerate(content_loader):
            # Get style batch (cycle if style dataset is smaller)
            try:
                style_batch = next(style_iter)
            except StopIteration:
                style_iter = iter(style_loader)
                style_batch = next(style_iter)

            content_batch = content_batch.to(device, non_blocking=True)
            style_batch = style_batch.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                content_features = encode_for_generator(
                    model, content_batch, args.train_encoder
                )
                style_features = encode_for_generator(
                    model, style_batch, args.train_encoder
                )
                blended_features = model.transform_features(
                    content_features, style_features, alpha=1.0
                )
                output = model.decoder(blended_features)

                with torch.no_grad():
                    loss_content_features = loss_encoder(content_batch)
                    loss_style_features = loss_encoder(style_batch)
                    loss_target_features = model.transform_features(
                        loss_content_features, loss_style_features, alpha=1.0
                    )

                output_features = loss_encoder(output)
                c_loss = compute_content_loss_from_features(
                    output_features, loss_target_features, content_feature_weights
                )
                s_loss = compute_style_loss_from_features(
                    output_features, loss_style_features, style_feature_weights
                )
                total_loss = args.content_weight * c_loss + args.style_weight * s_loss

            # Backward
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip,
                )

            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if not use_amp or scaler.get_scale() >= scale_before:
                optimizer_steps += 1

            # Track
            epoch_content_loss += c_loss.item()
            epoch_style_loss += s_loss.item()
            epoch_total_loss += total_loss.item()
            n_batches += 1

            if (batch_idx + 1) % args.log_interval == 0:
                elapsed = time.time() - t0
                print(
                    f"  Epoch [{epoch+1}/{args.epochs}] "
                    f"Batch [{batch_idx+1}/{len(content_loader)}] "
                    f"C_loss: {c_loss.item():.4f}  "
                    f"S_loss: {s_loss.item():.4f}  "
                    f"Total: {total_loss.item():.4f}  "
                    f"Time: {elapsed:.1f}s"
                )

            if args.max_steps_per_epoch and n_batches >= args.max_steps_per_epoch:
                break

        if optimizer_steps > 0:
            scheduler.step()

        # Epoch averages
        avg_c = epoch_content_loss / n_batches
        avg_s = epoch_style_loss / n_batches
        avg_t = epoch_total_loss / n_batches
        history["content_loss"].append(avg_c)
        history["style_loss"].append(avg_s)
        history["total_loss"].append(avg_t)

        epoch_time = time.time() - t0
        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Avg C_loss: {avg_c:.4f}  Avg S_loss: {avg_s:.4f}  "
            f"Avg Total: {avg_t:.4f}  Time: {epoch_time:.1f}s"
        )

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save(
                checkpoint_payload(
                    model, optimizer, epoch, avg_c, avg_s, args, loss_encoder
                ),
                ckpt_path,
            )
            print(f"  Saved checkpoint: {ckpt_path}")

        # Generate samples
        if (epoch + 1) % args.sample_interval == 0:
            generate_samples(
                model, content_loader, style_loader,
                device, epoch + 1, sample_dir, train_encoder=args.train_encoder
            )

    # Save final model
    final_path = ckpt_dir / "model_final.pt"
    torch.save(
        checkpoint_payload(
            model,
            optimizer,
            args.epochs - 1,
            history["content_loss"][-1],
            history["style_loss"][-1],
            args,
            loss_encoder,
        ),
        final_path,
    )

    # Save training history
    with open(ckpt_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Final model saved to {final_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet34-AdaIN Style Transfer")

    # Paths
    parser.add_argument("--content_dir", type=str, default="./data/coco/train2017")
    parser.add_argument("--style_dir", type=str, default="./data/wikiart")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--sample_dir", type=str, default="./samples")

    # Dataset
    parser.add_argument("--max_content_images", type=int, default=None)
    parser.add_argument("--max_style_images", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=256)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    # Loss
    parser.add_argument("--content_weight", type=float, default=1.0)
    parser.add_argument("--style_weight", type=float, default=10.0)
    parser.add_argument(
        "--content_feature_weights",
        type=str,
        default=None,
        help="Comma-separated weights for ResNet levels conv1..layer4.",
    )
    parser.add_argument(
        "--style_feature_weights",
        type=str,
        default=None,
        help="Comma-separated weights for ResNet levels conv1..layer4.",
    )

    # Optimizer
    parser.add_argument("--lr_encoder_low", type=float, default=1e-5)
    parser.add_argument("--lr_encoder_high", type=float, default=1e-4)
    parser.add_argument("--lr_decoder", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Scheduler
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step"])
    parser.add_argument("--T_max", type=int, default=50)

    # Depth
    parser.add_argument("--use_depth", action="store_true")
    parser.add_argument("--depth_scale", type=float, default=2.0)

    # Logging
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--sample_interval", type=int, default=1)

    # Misc
    parser.add_argument(
        "--pretrained_encoder",
        dest="pretrained_encoder",
        action="store_true",
        default=True,
    )
    parser.add_argument("--random_encoder", dest="pretrained_encoder", action="store_false")
    parser.add_argument("--train_encoder", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save_loss_encoder", action="store_true")
    parser.add_argument("--max_steps_per_epoch", type=int, default=None)
    parser.add_argument("--resume_from", type=str, default=None)

    # Quantization-aware training
    parser.add_argument("--qat", action="store_true", help="Enable ModelOpt fake-quant QAT.")
    parser.add_argument("--qat_mode", type=str, default="int8", choices=["int8", "int4"])
    parser.add_argument("--qat_calib_batches", type=int, default=32)
    parser.add_argument("--qat_algorithm", type=str, default="max")
    parser.add_argument("--qat_lr_scale", type=float, default=0.1)
    parser.add_argument("--qat_disable_encoder", action="store_true")
    parser.add_argument("--qat_disable_decoder", action="store_true")
    parser.add_argument("--qat_disable_final_layer", action="store_true", default=True)
    parser.add_argument(
        "--qat_quantize_final_layer",
        dest="qat_disable_final_layer",
        action="store_false",
        help="Allow quantization of the last RGB output convolution.",
    )
    parser.add_argument(
        "--qat_disable_patterns",
        type=str,
        default="",
        help="Comma-separated ModelOpt quantizer wildcard patterns to disable.",
    )
    parser.add_argument("--qat_print_summary", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
