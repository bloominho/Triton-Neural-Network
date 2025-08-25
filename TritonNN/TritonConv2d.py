import torch
import torch.nn as nn

from .kernel.Conv2d import triton_conv2d\

class TritonConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=False):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         1, 1, bias, 'zeros', 'cuda', torch.float32)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        if bias:
            raise NotImplementedError("Bias not supported yet")

    def forward(self, x):
        return triton_conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=(1, 1))