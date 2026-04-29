"""Adaptive Instance Normalization (AdaIN) layer."""

import torch
import torch.nn as nn


class AdaIN(nn.Module):
    """Adaptive Instance Normalization.
    
    Aligns the channel-wise mean and variance of content features
    to match those of style features. Core mechanism from:
    Huang & Belongie, "Arbitrary Style Transfer in Real-time with
    Adaptive Instance Normalization", ICCV 2017.
    
    AdaIN(x, y) = σ(y) * ((x - μ(x)) / σ(x)) + μ(y)
    """

    @staticmethod
    def calc_mean_std(
        feat: torch.Tensor, eps: float = 1e-5
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute channel-wise mean and std of feature maps."""
        n, c = feat.shape[:2]
        feat_flat = feat.view(n, c, -1)
        feat_mean = feat_flat.mean(dim=2).view(n, c, 1, 1)
        feat_var = feat_flat.var(dim=2) + eps
        feat_std = feat_var.sqrt().view(n, c, 1, 1)
        return feat_mean, feat_std

    def forward(
        self, content_feat: torch.Tensor, style_feat: torch.Tensor
    ) -> torch.Tensor:
        assert content_feat.shape[:2] == style_feat.shape[:2]
        size = content_feat.shape

        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)

        normalized = (content_feat - content_mean.expand(size)) / content_std.expand(
            size
        )
        return normalized * style_std.expand(size) + style_mean.expand(size)
