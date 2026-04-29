"""U-Net style decoder with skip connections for style transfer."""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """Decoder that reconstructs images from multi-level encoder features.
    
    Uses nearest-neighbor upsampling + conv blocks with skip connections
    from the encoder. InstanceNorm is used instead of BatchNorm to avoid
    washing out style information across the batch.
    
    Architecture:
        up4: 512 -> 256, concat with level 3 -> 512
        up3: 512 -> 128, concat with level 2 -> 256
        up2: 256 -> 64,  concat with level 1 -> 128
        up1: 128 -> 64
        final: 64 -> 32 -> 3
    """

    def __init__(self):
        super().__init__()

        self.up4 = self._up_block(512, 256)
        self.up3 = self._up_block(512, 128)  # 256 + 256 skip = 512
        self.up2 = self._up_block(256, 64)   # 128 + 128 skip = 256
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
        )

        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )

    @staticmethod
    def _up_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(True),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        x = features[-1]  # 512ch

        x = self.up4(x)
        x = torch.cat([x, features[-2]], dim=1)  # + 256ch skip

        x = self.up3(x)
        x = torch.cat([x, features[-3]], dim=1)  # + 128ch skip

        x = self.up2(x)
        x = torch.cat([x, features[-4]], dim=1)  # + 64ch skip

        x = self.up1(x)
        x = self.final(x)
        return x
