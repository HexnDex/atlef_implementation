"""
ATLEF Steganalysis: YeNet

Canonical YeNet (Ye, Ni and Yi, 2017): 30 trainable SRM high-pass
filters, a hard-tanh truncated linear unit (TLU), 8 convolutional
blocks, and a final linear classifier. This is the architecture
actually trained in Section 4.5 and used to produce the detection
results reported in Table 4.11 of the thesis. Matches Section 3.5.3
and Table 3.5.

Requires SRM_Kernels.npy, the 30-filter SRM bank used to initialize
the (trainable) preprocessing layer. Generate it once with
srm_kernels.py before training or loading a checkpoint.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

STEGO_IMG_CHANNEL = 1
STEGO_IMG_HEIGHT = 256  # patches are 256x256


class SRM_conv2d(nn.Module):
    """30 5x5 SRM filters, initialized from SRM_Kernels.npy.
    Trainable: the paper allows fine-tuning the initial filter bank."""

    def __init__(self, srm_kernels: np.ndarray, stride=1, padding=0):
        super().__init__()
        self.in_channels = STEGO_IMG_CHANNEL
        self.out_channels = 30
        self.kernel_size = (5, 5)
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (1, 1)
        self.groups = 1
        self.weight = Parameter(
            torch.Tensor(30, self.in_channels, 5, 5), requires_grad=True
        )
        self.bias = Parameter(torch.Tensor(30), requires_grad=True)
        self._srm_kernels = srm_kernels
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.numpy()[:] = self._srm_kernels
        self.bias.data.zero_()

    def forward(self, x):
        return F.conv2d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )


class ConvBlock(nn.Module):
    """Conv -> ReLU. Optional BatchNorm if with_bn=True."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, with_bn=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.relu = nn.ReLU()
        self.with_bn = with_bn
        if with_bn:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = lambda x: x
        self.reset_parameters()

    def forward(self, x):
        return self.norm(self.relu(self.conv(x)))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.conv.weight)
        self.conv.bias.data.fill_(0.2)
        if self.with_bn:
            self.norm.reset_parameters()


class YeNet(nn.Module):
    """Canonical YeNet: 30 SRM filters -> TLU(+-3) -> 8 conv blocks -> FC(N, 2).

    Input: (N, 1, 256, 256) grayscale image patches.
    Output: (N, 2) raw logits (cover, stego). Apply softmax for
    probabilities, or argmax for a binary prediction.
    """

    def __init__(self, srm_kernels: np.ndarray, with_bn=False, threshold=3):
        super().__init__()
        self.with_bn = with_bn
        self.preprocessing = SRM_conv2d(srm_kernels, stride=1, padding=0)
        self.TLU = nn.Hardtanh(-threshold, threshold, True)
        if with_bn:
            self.norm1 = nn.BatchNorm2d(30)
        else:
            self.norm1 = lambda x: x

        self.block2 = ConvBlock(30, 30, 3, with_bn=with_bn)
        self.block3 = ConvBlock(30, 30, 3, with_bn=with_bn)
        self.block4 = ConvBlock(30, 30, 3, with_bn=with_bn)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.block5 = ConvBlock(30, 32, 5, with_bn=with_bn)
        self.pool2 = nn.AvgPool2d(3, 2)
        self.block6 = ConvBlock(32, 32, 5, with_bn=with_bn)
        self.pool3 = nn.AvgPool2d(3, 2)
        self.block7 = ConvBlock(32, 32, 5, with_bn=with_bn)
        self.pool4 = nn.AvgPool2d(3, 2)
        self.block8 = ConvBlock(32, 16, 3, with_bn=with_bn)
        self.block9 = ConvBlock(16, 16, 3, 3, with_bn=with_bn)

        # FC input size depends on input image size: 256 -> 144 (16*3*3)
        self.num_of_neurons = 144 if STEGO_IMG_HEIGHT == 256 else 1024
        self.ip1 = nn.Linear(self.num_of_neurons, 2)
        self.reset_parameters()

    def forward(self, x):
        x = x.float()
        x = self.preprocessing(x)
        x = self.TLU(x)
        x = self.norm1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool1(x)
        x = self.block5(x)
        x = self.pool2(x)
        x = self.block6(x)
        x = self.pool3(x)
        x = self.block7(x)
        x = self.pool4(x)
        x = self.block8(x)
        x = self.block9(x)
        x = x.view(x.size(0), -1)
        x = self.ip1(x)
        return x

    def reset_parameters(self):
        for mod in self.modules():
            if isinstance(mod, (SRM_conv2d, nn.BatchNorm2d, ConvBlock)):
                mod.reset_parameters()
            elif isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, 0.0, 0.01)
                mod.bias.data.zero_()


if __name__ == "__main__":
    from srm_kernels import build_srm_kernels

    model = YeNet(build_srm_kernels())
    n_params = sum(p.numel() for p in model.parameters())
    dummy = torch.randn(2, 1, 256, 256)
    out = model(dummy)
    print(f"YeNet parameters: {n_params:,}")
    print(f"Output shape: {tuple(out.shape)}")
