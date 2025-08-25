import torch
import torch.nn as nn

import builtins
from .kernel.AvgPool2d import triton_avgpool2d 

class TritonAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride):
        super(TritonAvgPool2d, self).__init__()
        self.kernel_size = kernel_size 
        self.stride = stride

    def forward(self, x):
        return triton_avgpool2d(x, pool_size=(self.kernel_size, self.kernel_size), stride=(self.stride, self.stride))
