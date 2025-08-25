import torch
import torch.nn as nn

from .kernel.MaxPool2d import triton_maxpool2d


class TritonMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(TritonMaxPool2d, self).__init__()
        self.kernel_size = kernel_size 
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return triton_maxpool2d(x, kernel_size=(self.kernel_size, self.kernel_size), stride=(self.stride, self.stride), padding=(self.padding, self.padding))  