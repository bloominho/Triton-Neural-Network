import torch
import triton
import triton.language as tl

@triton.jit
def _linear_forward_kernel(
    x_ptr, w_ptr, b_ptr,
    out_ptr,
    M, N, K,
    x_stride_m, x_stride_k,
    w_stride_k, w_stride_n,
    out_stride_m, out_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr
):

    # Program ID (thread block ID)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # offs
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # accumulation, initialize to zero
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # RUN
    for k in range(0, K, BLOCK_K):
        index_K = offs_k + k
        index_x = offs_m[:, None] * x_stride_m + index_K[None, :] * x_stride_k
        x = tl.load(x_ptr + index_x, mask=(offs_m[:, None] < M) & (index_K[None, :] < K), other = 0.)
        
        index_w = index_K[:, None] * w_stride_k + offs_n[None, :] * w_stride_n
        w = tl.load(w_ptr + index_w, mask=(index_K[:, None] < K) & (offs_n[None, :] < N), other = 0.)
        
        acc += tl.dot(x, w)

    # Add Bias
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.).to(acc.dtype)
    acc += bias[None, :]
    
    index_out = offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n
    tl.store(out_ptr + index_out,
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def triton_linear(x, w, b):
    # Block Size
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    # Multiple Batches
    orig_shape = x.shape[:-1]
    if x.ndim > 2:
        x = x.view(-1, x.shape[-1])
    
    # Sizes of the Matrices
    M, K = x.shape
    K, N = w.shape
    
    # Output Array
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Triton Grid
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    
    _linear_forward_kernel[grid](
        x, w, b,
        out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N
    )

    # Recover original Batch
    return out.view(*orig_shape, N)