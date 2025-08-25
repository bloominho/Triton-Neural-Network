import torch
import torch.nn as nn

from .kernel.Linear import triton_linear

class TritonLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        act_func = None,
        device = 'cuda',
        dtype: torch.dtype = torch.float16,
        ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.act_func = act_func

    def forward(self, x):
        return triton_linear(x.to(torch.float16), self.weight.T.contiguous(), self.bias)