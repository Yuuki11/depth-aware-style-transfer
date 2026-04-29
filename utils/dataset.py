"""Dataset classes for style transfer training."""

import os
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageFolderDataset(Dataset):
    """Generic image folder dataset with optional max image limit.
    
    Recursively finds all images in a directory tree.
    
    Args:
        root: Root directory path.
        transform: Torchvision transforms to apply.
        max_images: Cap the number of images loaded (None = all).
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(
        self,
        root: str | Path,
        transform: transforms.Compose | None = None,
        max_images: int | None = None,
    ):
        self.root = Path(root)
        self.transform = transform

        self.paths = sorted([
            p for p in self.root.rglob("*")
            if p.suffix.lower() in self.EXTENSIONS
        ])

        if max_images is not None and max_images < len(self.paths):
            random.seed(42)
            self.paths = random.sample(self.paths, max_images)

        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img
        except Exception:
            # Return a random other image if this one is corrupted
            return self.__getitem__(random.randint(0, len(self) - 1))
