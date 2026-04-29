"""Calibration data helpers for PTQ and QAT."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import get_transform
from utils.dataset import ImageFolderDataset


def build_image_loaders(
    content_dir: str | Path,
    style_dir: str | Path,
    *,
    image_size: int,
    batch_size: int,
    max_content_images: int | None = None,
    max_style_images: int | None = None,
    num_workers: int = 0,
    shuffle: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Create paired content/style loaders using the repo's preprocessing."""
    transform = get_transform(image_size)
    content_dataset = ImageFolderDataset(
        content_dir,
        transform=transform,
        max_images=max_content_images,
    )
    style_dataset = ImageFolderDataset(
        style_dir,
        transform=transform,
        max_images=max_style_images,
    )
    content_loader = DataLoader(
        content_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    style_loader = DataLoader(
        style_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return content_loader, style_loader


def paired_batches(
    content_loader: DataLoader,
    style_loader: DataLoader,
    *,
    max_batches: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield content/style batches, cycling the shorter loader."""
    style_iter = cycle(style_loader)
    for idx, content in enumerate(content_loader):
        if idx >= max_batches:
            break
        yield content, next(style_iter)


@torch.no_grad()
def run_calibration_forward_loop(
    model: torch.nn.Module,
    content_loader: DataLoader,
    style_loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int,
) -> None:
    """Forward representative batches through a two-input style-transfer model."""
    was_training = model.training
    model.eval()
    for content, style in paired_batches(content_loader, style_loader, max_batches=max_batches):
        model(content.to(device, non_blocking=True), style.to(device, non_blocking=True))
    model.train(was_training)


def collect_onnx_calibration_arrays(
    content_loader: DataLoader,
    style_loader: DataLoader,
    *,
    max_batches: int,
) -> dict[str, np.ndarray]:
    """Collect calibration tensors for ModelOpt ONNX PTQ."""
    content_batches = []
    style_batches = []
    for content, style in paired_batches(content_loader, style_loader, max_batches=max_batches):
        content_batches.append(content.cpu().numpy().astype(np.float32, copy=False))
        style_batches.append(style.cpu().numpy().astype(np.float32, copy=False))
    if not content_batches:
        raise RuntimeError("No calibration batches were produced.")
    return {
        "content": np.concatenate(content_batches, axis=0),
        "style": np.concatenate(style_batches, axis=0),
    }


@torch.no_grad()
def collect_depth_onnx_calibration_arrays(
    depth_model: torch.nn.Module,
    content_loader: DataLoader,
    style_loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int,
) -> dict[str, np.ndarray]:
    """Collect RGB and explicit depth tensors for depth-aware ONNX calibration."""
    content_batches = []
    style_batches = []
    content_depth_batches = []
    style_depth_batches = []
    depth_model.eval()

    for content, style in paired_batches(content_loader, style_loader, max_batches=max_batches):
        content = content.to(device, non_blocking=True)
        style = style.to(device, non_blocking=True)
        content_depth = depth_model.depth_estimator.from_tensor(content).to(device)
        style_depth = depth_model.depth_estimator.from_tensor(style).to(device)
        content_batches.append(content.cpu().numpy().astype(np.float32, copy=False))
        style_batches.append(style.cpu().numpy().astype(np.float32, copy=False))
        content_depth_batches.append(content_depth.cpu().numpy().astype(np.float32, copy=False))
        style_depth_batches.append(style_depth.cpu().numpy().astype(np.float32, copy=False))

    if not content_batches:
        raise RuntimeError("No depth calibration batches were produced.")

    return {
        "content": np.concatenate(content_batches, axis=0),
        "content_depth": np.concatenate(content_depth_batches, axis=0),
        "style": np.concatenate(style_batches, axis=0),
        "style_depth": np.concatenate(style_depth_batches, axis=0),
    }
