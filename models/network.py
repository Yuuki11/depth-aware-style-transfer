"""Complete style transfer networks."""

import torch
import torch.nn as nn

from .encoder import ResNetEncoder, DepthAwareResNetEncoder
from .decoder import Decoder
from .adain import AdaIN
from .depth import DepthEstimator


class StyleTransferNet(nn.Module):
    """AdaIN-based style transfer with ResNet34 encoder.
    
    Architecture:
        1. Encode content and style images → multi-level features
        2. Apply AdaIN at each feature level to blend statistics
        3. Decode blended features → stylized output
    
    Args:
        pretrained_encoder: Use ImageNet pretrained weights.
    """

    def __init__(self, pretrained_encoder: bool = False):
        super().__init__()
        self.encoder = ResNetEncoder(pretrained=pretrained_encoder)
        self.decoder = Decoder()
        self.adain = AdaIN()

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.encoder(x)

    def transform_features(
        self,
        content_features: list[torch.Tensor],
        style_features: list[torch.Tensor],
        alpha: float = 1.0,
    ) -> list[torch.Tensor]:
        """Apply AdaIN to matching content/style feature pyramids."""
        blended = []
        for cf, sf in zip(content_features, style_features):
            t = self.adain(cf, sf)
            blended.append(alpha * t + (1 - alpha) * cf)
        return blended

    def forward(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            content: Content image tensor (B, 3, H, W).
            style: Style image tensor (B, 3, H, W).
            alpha: Style strength. 0 = content only, 1 = full style.
            
        Returns:
            Stylized image tensor (B, 3, H, W).
        """
        content_features = self.encoder(content)
        style_features = self.encoder(style)

        return self.decoder(self.transform_features(content_features, style_features, alpha))


class DepthAwareStyleTransferNet(nn.Module):
    """Depth-aware style transfer with 4-channel ResNet34 encoder.
    
    Extends StyleTransferNet by concatenating Depth Anything V2 
    pseudo-depth maps with RGB input, allowing the encoder to learn
    depth-aware features for better spatial coherence.
    
    Args:
        depth_scale: Multiplier for depth channel input.
        pretrained_encoder: Use ImageNet pretrained weights for RGB channels.
        depth_model_name: HuggingFace model ID for depth estimation.
    """

    def __init__(
        self,
        depth_scale: float = 2.0,
        pretrained_encoder: bool = False,
        depth_model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
    ):
        super().__init__()
        self.encoder = DepthAwareResNetEncoder(
            pretrained=pretrained_encoder, depth_scale=depth_scale
        )
        self.decoder = Decoder()
        self.adain = AdaIN()
        self._depth_estimator = None
        self._depth_model_name = depth_model_name

    @property
    def depth_estimator(self) -> DepthEstimator:
        """Lazy-init depth estimator on first use."""
        if self._depth_estimator is None:
            device = next(self.parameters()).device
            self._depth_estimator = DepthEstimator(
                model_name=self._depth_model_name,
                device=str(device),
            )
        return self._depth_estimator

    def load_rgb_checkpoint(
        self, checkpoint_path: str, device: torch.device
    ) -> None:
        """Load weights from an RGB-only StyleTransferNet checkpoint.
        
        Handles the 3→4 channel conv1 weight conversion by initializing
        the depth channel from the mean of RGB weights.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]

        new_state_dict = {}
        for k, v in state_dict.items():
            if k == "encoder.conv1.weight":
                if v.shape[1] == 4:
                    new_state_dict[k] = v
                else:
                    new_w = torch.zeros(64, 4, 7, 7, device=device)
                    new_w[:, :3] = v
                    new_w[:, 3:] = v.mean(dim=1, keepdim=True) * 0.5
                    new_state_dict[k] = new_w
            else:
                new_state_dict[k] = v

        self.load_state_dict(new_state_dict)

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        depth = self.depth_estimator.from_tensor(x)
        return self.encoder(x, depth)

    def transform_features(
        self,
        content_features: list[torch.Tensor],
        style_features: list[torch.Tensor],
        alpha: float = 1.0,
    ) -> list[torch.Tensor]:
        """Apply AdaIN to matching depth-aware feature pyramids."""
        blended = []
        for cf, sf in zip(content_features, style_features):
            t = self.adain(cf, sf)
            blended.append(alpha * t + (1 - alpha) * cf)
        return blended

    def forward(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        content_depth = self.depth_estimator.from_tensor(content)
        style_depth = self.depth_estimator.from_tensor(style)

        content_features = self.encoder(content, content_depth)
        style_features = self.encoder(style, style_depth)
        return self.decoder(self.transform_features(content_features, style_features, alpha))
