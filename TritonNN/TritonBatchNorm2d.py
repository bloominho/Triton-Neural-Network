import torch
import torch.nn as nn

from .kernel.BatchNorm2d import triton_bn2d

class TritonBatchNorm2d(nn.BatchNorm2d):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        act_func = None,
        device = 'cuda',
        dtype: torch.dtype = torch.float32,
        ) -> None:
        super().__init__(num_features, eps, momentum, affine,
                         track_running_stats, device, dtype)
        self.act_func = act_func

    def forward(self, input):
        self._check_input_dim(input)
        return triton_bn2d(input,
                                       self.weight, self.bias,
                                       self.running_mean, self.running_var,
                                       self.momentum, self.eps)
