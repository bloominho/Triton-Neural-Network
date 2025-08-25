import torch
import triton
import triton.language as tl

@triton.jit
def _bn2d_forward_kernel(
    in_ptr, # Input
    out_ptr, # Output
    scale_ptr, shift_ptr, # Parameters
    N, C, H, W, # Input Size
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offs < N * C * H * W

    # Calculate Channel
    C_index = (offs // (H * W)) % C

    # Load Data
    val = tl.load(in_ptr + offs, mask=mask, other=0.)
    scale = tl.load(scale_ptr + C_index, mask=mask, other=0.)
    shift = tl.load(shift_ptr + C_index, mask=mask, other=0.)

    # Calculate Norm
    output = val * scale + shift

    tl.store(out_ptr + offs, output.to(tl.float32), mask=mask)

def triton_bn2d(x,
                weight, bias,
                running_mean, running_var,
                momentum, eps):
    N, C, H, W = x.shape

    # Output
    out = torch.empty(x.shape, device = x.device, dtype=x.dtype)

    # Grid
    def grid(meta):
        return (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )

    # Parameters
    scale = ((running_var + eps).rsqrt() * weight).contiguous()
    shift = (bias - running_mean * scale).contiguous()

    # Launch Kernel
    _bn2d_forward_kernel[grid](
        x,
        out,
        scale, shift,
        N, C, H, W,
        BLOCK_SIZE=1024
    )
    
    return out
