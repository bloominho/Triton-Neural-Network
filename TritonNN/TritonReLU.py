import torch
import torch.nn as nn

from .kernel.ReLU import triton_relu 

class TritonReLU(nn.Module):
    def __init__(self, inplace=False):
        super(TritonReLU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return triton_relu(x, self.inplace)
