"""Lightweight student network used for MIT Indoor-67 classification."""

import torch
from torch import nn


class SqueezeExcitation(nn.Module):
    """Reweight feature channels using global image information."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden_channels = max(8, channels // reduction)

        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.layers(inputs)


class DSResidualBlock(nn.Module):
    """Depthwise-separable residual block with squeeze-excitation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        hidden_channels = in_channels * expansion
        self.use_residual = stride == 1 and in_channels == out_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            SqueezeExcitation(hidden_channels),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.block(inputs)
        return inputs + outputs if self.use_residual else outputs


class EfficientSceneStudent(nn.Module):
    """Compact CNN distilled from the ConvNeXt teacher model."""

    def __init__(self, num_classes: int = 67, width: float = 1.0) -> None:
        super().__init__()
        channels = [int(channel * width) for channel in (32, 48, 80, 128, 192)]

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                channels[0],
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(inplace=True),
        )

        self.features = nn.Sequential(
            DSResidualBlock(channels[0], channels[1], stride=2, expansion=2),
            DSResidualBlock(channels[1], channels[1], stride=1, expansion=2),
            DSResidualBlock(channels[1], channels[2], stride=2, expansion=3),
            DSResidualBlock(channels[2], channels[2], stride=1, expansion=3),
            DSResidualBlock(channels[2], channels[3], stride=2, expansion=3),
            DSResidualBlock(channels[3], channels[3], stride=1, expansion=3),
            DSResidualBlock(channels[3], channels[4], stride=2, expansion=4),
            DSResidualBlock(channels[4], channels[4], stride=1, expansion=4),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(channels[4], num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stem(inputs)
        features = self.features(features)
        return self.classifier(features)
