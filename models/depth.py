"""Depth estimation using Depth Anything V2."""

import torch
from PIL import Image


class DepthEstimator:
    """Wrapper around Depth Anything V2 for monocular depth estimation.
    
    Generates normalized pseudo-depth maps from RGB images using the
    pretrained Depth Anything V2 Small model from HuggingFace.
    
    Depth maps are normalized to [0, 1] and a power transform (γ=0.75)
    is applied to enhance depth contrast.
    
    Args:
        model_name: HuggingFace model identifier.
        device: Torch device to run inference on.
    """

    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: str = "cuda",
    ):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = device
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)
        self.model.eval()
        self.size = (
            self.image_processor.size["height"],
            self.image_processor.size["width"],
        )
        self.mean = torch.tensor(self.image_processor.image_mean).view(1, 3, 1, 1)
        self.std = torch.tensor(self.image_processor.image_std).view(1, 3, 1, 1)

    @staticmethod
    def _normalize_depth(depth: torch.Tensor) -> torch.Tensor:
        """Normalize predicted depth independently per image."""
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)

        flat = depth.flatten(start_dim=1)
        depth_min = flat.min(dim=1).values.view(-1, 1, 1)
        depth_max = flat.max(dim=1).values.view(-1, 1, 1)
        depth = (depth - depth_min) / (depth_max - depth_min + 1e-8)
        depth = torch.pow(depth, 0.75)
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        return depth

    @torch.no_grad()
    def estimate(self, image: Image.Image | list[Image.Image]) -> torch.Tensor:
        """Estimate depth from one PIL image or a batch of PIL images.
        
        Args:
            image: RGB PIL image.
            
        Returns:
            Depth map tensor of shape (B, 1, H, W), normalized [0, 1].
        """
        inputs = self.image_processor(images=image, return_tensors="pt")
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            outputs = self.model(**inputs)
            depth = outputs.predicted_depth

        return self._normalize_depth(depth)

    def from_tensor(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Estimate depth from a normalized image tensor.
        
        Denormalizes and preprocesses on-device, avoiding PIL/CPU conversion.
        
        Args:
            img_tensor: Tensor of shape (B, 3, H, W), ImageNet-normalized.
            
        Returns:
            Depth map tensor of shape (B, 1, H, W).
        """
        original_size = img_tensor.shape[-2:]
        device = img_tensor.device
        mean = self.mean.to(device=device, dtype=img_tensor.dtype)
        std = self.std.to(device=device, dtype=img_tensor.dtype)
        denorm = (img_tensor * std + mean).clamp(0, 1)
        pixel_values = torch.nn.functional.interpolate(
            denorm.float(), size=self.size, mode="bicubic", align_corners=False
        )
        pixel_values = (pixel_values - self.mean.to(device=device)) / self.std.to(device=device)

        device_type = "cuda" if str(device).startswith("cuda") else "cpu"
        with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=False):
            outputs = self.model(pixel_values=pixel_values)
            depth = self._normalize_depth(outputs.predicted_depth)

        if depth.shape[-2:] != original_size:
            depth = torch.nn.functional.interpolate(
                depth, size=original_size, mode="bilinear", align_corners=False
            )
        return depth
