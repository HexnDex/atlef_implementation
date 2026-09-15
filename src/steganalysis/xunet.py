"""
ATLEF Steganalysis: XuNet

Canonical XuNet (Xu, Wu and Shi, 2016): a single trainable KV high-pass
filter, five convolutional blocks, global average pooling, and a
two-layer classifier head. This is the architecture actually trained
in Section 4.5 and used to produce the detection results reported in
Table 4.11 of the thesis. Matches Section 3.5.3 and Table 3.5.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

STEGO_IMG_CHANNEL = 1


class ImageProcessing(nn.Module):
    """Single KV high-pass 5x5 filter applied to the input.
    The filter is a trainable nn.Parameter, initialized from the
    published KV values, but free to adapt during training."""

    def __init__(self) -> None:
        super().__init__()
        self.kv_filter = nn.Parameter(
            torch.tensor(
                [[-1.0, 2.0, -2.0, 2.0, -1.0],
                 [2.0, -6.0, 8.0, -6.0, 2.0],
                 [-2.0, 8.0, -12.0, 8.0, -2.0],
                 [2.0, -6.0, 8.0, -6.0, 2.0],
                 [-1.0, 2.0, -2.0, 2.0, -1.0]],
            ).view(1, 1, 5, 5) / 12.0
        )

    def forward(self, inp: Tensor) -> Tensor:
        # input is (N, 1, H, W); apply the KV filter to the single channel
        for i in range(STEGO_IMG_CHANNEL):
            ch = inp[:, i, :, :].unsqueeze(dim=1)
            conv = F.conv2d(ch, self.kv_filter, stride=1, padding=2)
            features = conv if i == 0 else torch.cat((features, conv), dim=1)
        return features


class ConvBlock(nn.Module):
    """Conv -> (optional abs) -> BatchNorm -> activation -> AvgPool 5x5 stride 2."""

    def __init__(self, in_channels, out_channels, kernel_size, activation="relu", abs=False):
        super().__init__()
        self.padding = 2 if kernel_size == 5 else 0
        self.activation = nn.Tanh() if activation == "tanh" else nn.ReLU()
        self.abs = abs
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=1, padding=self.padding, bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)

    def forward(self, inp: Tensor) -> Tensor:
        out = self.conv(inp)
        if self.abs:
            out = torch.abs(out)
        out = self.batch_norm(out)
        out = self.activation(out)
        return self.pool(out)


class XuNet(nn.Module):
    """Canonical XuNet:
    KV filter -> Conv5x5/Abs/TanH -> Conv5x5/TanH -> 3 x Conv1x1/ReLU
    -> global average pool -> FC(128 -> 128 -> 2).

    Input: (N, 1, H, W) grayscale image patches.
    Output: (N, 2) raw logits (cover, stego). Apply softmax for
    probabilities, or argmax for a binary prediction.
    """

    def __init__(self):
        super().__init__()
        self.ImageProcessingLayer = ImageProcessing()
        self.layer1 = ConvBlock(STEGO_IMG_CHANNEL, 8, kernel_size=5, activation="tanh", abs=True)
        self.layer2 = ConvBlock(8, 16, kernel_size=5, activation="tanh")
        self.layer3 = ConvBlock(16, 32, kernel_size=1)
        self.layer4 = ConvBlock(32, 64, kernel_size=1)
        self.layer5 = ConvBlock(64, 128, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.fully_connected = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, image: Tensor) -> Tensor:
        with torch.no_grad():
            out = self.ImageProcessingLayer(image)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        return self.fully_connected(out)


if __name__ == "__main__":
    model = XuNet()
    n_params = sum(p.numel() for p in model.parameters())
    dummy = torch.randn(2, 1, 256, 256)
    out = model(dummy)
    print(f"XuNet parameters: {n_params:,}")
    print(f"Output shape: {tuple(out.shape)}")
