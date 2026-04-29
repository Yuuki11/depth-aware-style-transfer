"""Shared utilities for image loading, saving, and normalization."""

import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from pathlib import Path

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transform(size: int = 256) -> transforms.Compose:
    """Standard transform: resize, center crop, normalize."""
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_train_transform(size: int = 256) -> transforms.Compose:
    """Training transform with augmentation."""
    return transforms.Compose([
        transforms.Resize(size + 32),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_image(path: str | Path, size: int = 256) -> torch.Tensor:
    """Load image, transform, and return as (1, 3, H, W) tensor."""
    image = Image.open(path).convert("RGB")
    return get_transform(size)(image).unsqueeze(0)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization. Returns tensor in [0, 1]."""
    device = tensor.device
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def save_image(tensor: torch.Tensor, path: str | Path) -> None:
    """Denormalize and save a single image tensor."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(denormalize(tensor), str(path))


def save_comparison_grid(
    content: torch.Tensor,
    style: torch.Tensor,
    output: torch.Tensor,
    path: str | Path,
    depth_output: torch.Tensor | None = None,
) -> None:
    """Save a side-by-side comparison grid."""
    images = [denormalize(content), denormalize(style), denormalize(output)]
    if depth_output is not None:
        images.append(denormalize(depth_output))
    
    grid = torch.cat(images, dim=0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(grid, str(path), nrow=len(images), padding=4)
