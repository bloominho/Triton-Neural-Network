import torch
from .Linear import triton_linear

def im2col(x,
            H_kernel, W_kernel,
            H_stride, W_stride,
            H_padding, W_padding,
            H_dilation, W_dilation):
    N, C, H, W = x.shape
    
    # Zero Padding
    H_padded = H + 2 * H_padding
    W_padded = W + 2 * W_padding
    x_padded = x.new_zeros((N, C, H_padded, W_padded))
    x_padded[:, :, H_padding:H_padding+H, W_padding:W_padding+W] = x
    
    # Output Size
    H_out = (H_padded - (H_dilation * (H_kernel - 1) + 1)) // H_stride + 1
    W_out = (W_padded - (W_dilation * (W_kernel - 1) + 1)) // W_stride + 1
    
    # Reorgranize using as_strided
    shape = (N, H_out, W_out, C, H_kernel, W_kernel)
    strides = (
        x_padded.stride(0),
        x_padded.stride(2) * H_stride,
        x_padded.stride(3) * W_stride,
        x_padded.stride(1),
        x_padded.stride(2) * H_dilation,
        x_padded.stride(3) * W_dilation
    )
    col = x_padded.as_strided(shape, strides)
    
    # Reorder : Where the actually copying occurs
    col = col.reshape(N, H_out*W_out, C*H_kernel*W_kernel).contiguous()
    return col
    
def triton_conv2d(x, weight, bias, stride, padding, dilation):
    N, C, H, W = x.shape
    C_out, C, H_kernel, W_kernel = weight.shape
    H_stride, W_stride = stride
    H_padding, W_padding = padding
    H_dilation, W_dilation = dilation

    # Im2Col
    col = im2col(x, H_kernel, W_kernel, H_stride, W_stride, H_padding, W_padding, H_dilation, W_dilation)
    N, P, K = col.shape # K: C*H_Kernel*W_kernel, P: H_out * W_out
    
    # Reshape: So that it's 2D
    X_col_2d = col.reshape(-1, K)
    
    # 3. weight reshape
    W_col_2d = weight.view(C_out, K).t().contiguous()
    
    # 4. GEMM
    out_2d = triton_linear(X_col_2d, W_col_2d, torch.zeros(weight.shape[0], device=x.device, dtype=x.dtype))
    
    # 5. Reshape Output
    H_out = (H + 2 * H_padding - H_dilation * (H_kernel - 1) - 1) // H_stride + 1
    W_out = (W + 2 * W_padding - W_dilation * (W_kernel - 1) - 1) // W_stride + 1
    
    out = out_2d.view(N, H_out, W_out, C_out).permute(0, 3, 1, 2).contiguous()
    return out