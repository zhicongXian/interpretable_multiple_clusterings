from torch import Tensor, nn
import torch.nn.functional as F
from typing import Any, Iterable, Sequence
import math
import torch

def _group_count(channels: int, maximum_groups: int = 8) -> int:
    for groups in range(min(maximum_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _normalization(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(_group_count(channels), channels)

class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = _normalization(output_channels)
        self.conv2 = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = _normalization(output_channels)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = (
            nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                _normalization(output_channels),
            )
            if stride != 1 or input_channels != output_channels
            else nn.Identity()
        )

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.skip(inputs)
        output = self.activation(self.norm1(self.conv1(inputs)))
        output = self.dropout(output)
        output = self.norm2(self.conv2(output))
        return self.activation(output + residual)


class ResidualUpBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm1 = _normalization(output_channels)
        self.conv2 = nn.Conv2d(
            output_channels, output_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = _normalization(output_channels)
        self.skip = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
            _normalization(output_channels),
        )
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = F.interpolate(inputs, scale_factor=2, mode="bilinear", align_corners=False)
        residual = self.skip(inputs)
        output = self.activation(self.norm1(self.conv1(inputs)))
        output = self.dropout(output)
        output = self.norm2(self.conv2(output))
        return self.activation(output + residual)


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        channels: Sequence[int],
        blocks_per_stage: Sequence[int],
        latent_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError("input_shape must be (channels, height, width).")
        if not channels or len(channels) != len(blocks_per_stage):
            raise ValueError("Encoder channels and block counts must have equal length.")

        self.input_shape = tuple(int(value) for value in input_shape)
        first_channels = int(channels[0])
        self.stem = nn.Sequential(
            nn.Conv2d(
                self.input_shape[0],
                first_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _normalization(first_channels),
            nn.SiLU(),
        )
        stages: list[nn.Module] = []
        current_channels = first_channels
        for stage_index, (stage_channels, block_count) in enumerate(
            zip(channels, blocks_per_stage)
        ):
            stage_channels = int(stage_channels)
            stage_blocks: list[nn.Module] = [
                ResidualBlock(
                    current_channels,
                    stage_channels,
                    stride=1 if stage_index == 0 else 2,
                    dropout=dropout,
                )
            ]
            stage_blocks.extend(
                ResidualBlock(stage_channels, stage_channels, dropout=dropout)
                for _ in range(int(block_count) - 1)
            )
            stages.append(nn.Sequential(*stage_blocks))
            current_channels = stage_channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.to_latent = nn.Linear(current_channels, latent_dim)
        self.activation = nn.ReLU()
        self.latent_norm = nn.LayerNorm(latent_dim)

    def forward(self, images: Tensor) -> Tensor:
        features = self.stages(self.stem(images))
        return self.latent_norm(self.to_latent(self.pool(features).flatten(1)))


class ResNetDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        output_shape: Sequence[int],
        channels: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        if len(output_shape) != 3 or len(channels) < 2:
            raise ValueError("Decoder requires output (C,H,W) and at least two stages.")
        self.output_shape = tuple(int(value) for value in output_shape)
        self.channels = tuple(int(value) for value in channels)
        upsampling_factor = 2 ** (len(self.channels) - 1)
        self.base_height = max(2, math.ceil(self.output_shape[1] / upsampling_factor))
        self.base_width = max(2, math.ceil(self.output_shape[2] / upsampling_factor))
        self.from_latent = nn.Linear(
            latent_dim,
            self.channels[0] * self.base_height * self.base_width,
        )
        self.blocks = nn.Sequential(
            *[
                ResidualUpBlock(left, right, dropout=dropout)
                for left, right in zip(self.channels[:-1], self.channels[1:])
            ]
        )
        self.to_image = nn.Conv2d(
            self.channels[-1], self.output_shape[0], kernel_size=3, padding=1
        )

    def forward(self, latent: Tensor) -> Tensor:
        output = self.from_latent(latent).view(
            latent.shape[0], self.channels[0], self.base_height, self.base_width
        )
        output = self.blocks(output)
        if tuple(output.shape[-2:]) != self.output_shape[1:]:
            output = F.interpolate(
                output,
                size=self.output_shape[1:],
                mode="bilinear",
                align_corners=False,
            )
        return torch.sigmoid(self.to_image(output))