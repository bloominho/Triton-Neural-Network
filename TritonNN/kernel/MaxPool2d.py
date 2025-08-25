import torch
import triton
import triton.language as tl

@triton.jit
def _maxpool2d_forward_kernel(
    in_ptr, # Input
    out_ptr, # Output
    N, C, H, W, # Input Size
    H_out, W_out, # Output Size
    H_kernel, W_kernel, # Kernel Size
    H_stride, W_stride, # Stride Size
    H_padding, W_padding, # Padding Size
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask = offs < N * C * H_out * W_out

    # Map this program to an output element
    n = offs // (C * H_out * W_out)
    temp = offs % (C * H_out * W_out)
    c = temp // (H_out * W_out)
    temp = temp % (H_out * W_out)
    h = temp // W_out
    w = temp % W_out
    
    image_offset = (C * n + c) * H * W
    
    # max
    max = tl.full((BLOCK_SIZE,), -float("inf"), dtype=tl.float32)
    
    # Pooling Window
    for kH in range(H_kernel):
        for kW in range(W_kernel):
            index_H = h * H_stride + kH - H_padding
            index_W = w * W_stride + kW - W_padding
            boundary_check = mask & (index_H >= 0) & (index_H < H) & (index_W >= 0) & (index_W < W)
            val = tl.load(in_ptr + image_offset + index_H * W + index_W, mask=boundary_check, other=-float("inf"))
            max = tl.maximum(max, val)
    
    tl.store(out_ptr + offs, max, mask = mask)

def triton_maxpool2d(x, kernel_size, stride, padding):
    N, C, H, W = x.shape
    H_kernel, W_kernel = kernel_size
    H_stride, W_stride = stride
    H_padding, W_padding = padding
    H_out = (H + 2*H_padding - H_kernel) // H_stride + 1
    W_out = (W + 2*W_padding - W_kernel) // W_stride + 1
    
    x_flat = x.contiguous().view(-1)
    out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)
    out_flat = out.view(-1)
    
    def grid(meta):
        B = meta['BLOCK_SIZE']
        return ((N*C*H_out*W_out + B - 1) // B ,)
    
    _maxpool2d_forward_kernel[grid](
        x_flat, out_flat,
        N, C, H, W,
        H_out, W_out,
        H_kernel, W_kernel,
        H_stride, W_stride,
        H_padding, W_padding,
        BLOCK_SIZE=1024
    )
    
    return out
