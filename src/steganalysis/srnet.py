"""
ATLEF Steganalysis: SRNet

Canonical 12-layer residual CNN (Boroumand, Chen and Fridrich, 2019).
Unlike YeNet and XuNet, SRNet has no fixed SRM prefilter, it learns its
own front-end directly from raw pixel input. This is the architecture
actually trained in Section 4.5 and used to produce the detection
results reported in Table 4.11 of the thesis. Matches Section 3.5.3
and Table 3.5.

Output head note: the canonical paper uses Linear(512, 2) with
log-softmax. This implementation uses Linear(512, 1) with a single
logit so the same BCEWithLogitsLoss training loop can be shared with
YeNet and XuNet. Mathematically equivalent for binary classification.
"""

import torch
import torch.nn as nn
from torch import Tensor


class ConvBn(nn.Module):
    """Conv 3x3, stride 1, padding 1, followed by BatchNorm."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=True,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, inp: Tensor) -> Tensor:
        return self.batch_norm(self.conv(inp))


class Type1(nn.Module):
    """ConvBn + ReLU. No pooling, no skip connection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.convbn = ConvBn(in_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, inp: Tensor) -> Tensor:
        return self.relu(self.convbn(inp))


class Type2(nn.Module):
    """Residual block: input + ConvBn(Type1(input))."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.type1 = Type1(in_channels, out_channels)
        self.convbn = ConvBn(in_channels, out_channels)

    def forward(self, inp: Tensor) -> Tensor:
        return inp + self.convbn(self.type1(inp))


class Type3(nn.Module):
    """Residual block with stride-2 downsampling.
    main path: AvgPool(ConvBn(Type1(input)))
    skip path: BatchNorm(Conv1x1 stride 2(input))
    output = main + skip
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=2, padding=0, bias=True,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.type1 = Type1(in_channels, out_channels)
        self.convbn = ConvBn(out_channels, out_channels)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, inp: Tensor) -> Tensor:
        out = self.batch_norm(self.conv1(inp))
        out1 = self.pool(self.convbn(self.type1(inp)))
        return out + out1


class Type4(nn.Module):
    """Final block: Type1 -> ConvBn -> global average pool."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.type1 = Type1(in_channels, out_channels)
        self.convbn = ConvBn(out_channels, out_channels)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, inp: Tensor) -> Tensor:
        return self.gap(self.convbn(self.type1(inp)))


class Srnet(nn.Module):
    """Real SRNet (Boroumand, Chen and Fridrich, 2019).

    Input: (N, 1, H, W) grayscale image patches.
    Output: (N, 1) raw logits. Apply sigmoid for a stego probability,
    or threshold at 0 for a binary prediction.
    """

    def __init__(self) -> None:
        super().__init__()
        self.type1s = nn.Sequential(
            Type1(1, 64),
            Type1(64, 16),
        )
        self.type2s = nn.Sequential(
            Type2(16, 16),
            Type2(16, 16),
            Type2(16, 16),
            Type2(16, 16),
            Type2(16, 16),
        )
        self.type3s = nn.Sequential(
            Type3(16, 16),
            Type3(16, 64),
            Type3(64, 128),
            Type3(128, 256),
        )
        self.type4 = Type4(256, 512)
        self.fc = nn.Linear(512, 1)  # BCEWithLogitsLoss-compatible

    def forward(self, inp: Tensor) -> Tensor:
        out = self.type1s(inp)
        out = self.type2s(out)
        out = self.type3s(out)
        out = self.type4(out)
        out = out.view(out.size(0), -1)  # (N, 512)
        return self.fc(out)  # (N, 1) logits


if __name__ == "__main__":
    model = Srnet()
    n_params = sum(p.numel() for p in model.parameters())
    dummy = torch.randn(2, 1, 256, 256)
    out = model(dummy)
    print(f"Srnet parameters: {n_params:,}")
    print(f"Output shape: {tuple(out.shape)}")
