"""ResNet-based encoders for style transfer."""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    """ResNet34 encoder that extracts multi-level features for style transfer.
    
    Replaces VGG19 encoder from original AdaIN. Extracts 5 feature levels
    from conv1 through layer4, providing hierarchical representations
    from edges/textures (early) to semantic content (deep).
    
    Feature levels:
        0: after conv1+bn+relu  — 64 channels
        1: after layer1         — 64 channels  
        2: after layer2         — 128 channels
        3: after layer3         — 256 channels
        4: after layer4         — 512 channels
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()
        resnet = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)

        # Use stride=1 in conv1 to preserve spatial resolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        if pretrained:
            self.conv1.weight.data.copy_(resnet.conv1.weight.data)

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)  # Level 0: 64ch

        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)  # Level 1: 64ch

        x = self.layer2(x)
        features.append(x)  # Level 2: 128ch

        x = self.layer3(x)
        features.append(x)  # Level 3: 256ch

        x = self.layer4(x)
        features.append(x)  # Level 4: 512ch

        return features


class DepthAwareResNetEncoder(nn.Module):
    """ResNet34 encoder with 4-channel input (RGB + depth).
    
    Accepts concatenated [RGB, depth] input. The depth channel weight
    is initialized from the mean of RGB weights scaled by 0.5.
    
    Args:
        pretrained: Use ImageNet pretrained weights for RGB channels.
        depth_scale: Multiplier applied to depth map before concatenation.
    """

    def __init__(self, pretrained: bool = False, depth_scale: float = 2.0):
        super().__init__()
        resnet = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)

        # 4-channel conv1: 3 RGB + 1 depth
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=1, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight[:, :3] = resnet.conv1.weight
            self.conv1.weight[:, 3:] = resnet.conv1.weight.mean(dim=1, keepdim=True) * 0.5

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.depth_scale = depth_scale

    def forward(
        self, x: torch.Tensor, depth_map: torch.Tensor | None = None
    ) -> list[torch.Tensor]:
        if depth_map is not None:
            depth_map = depth_map.to(x.device) * self.depth_scale
            if depth_map.shape[-2:] != x.shape[-2:]:
                depth_map = nn.functional.interpolate(
                    depth_map, size=x.shape[-2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, depth_map], dim=1)
        else:
            zero_depth = torch.zeros(
                (x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device
            )
            x = torch.cat([x, zero_depth], dim=1)

        features = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)

        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)

        x = self.layer2(x)
        features.append(x)

        x = self.layer3(x)
        features.append(x)

        x = self.layer4(x)
        features.append(x)

        return features
