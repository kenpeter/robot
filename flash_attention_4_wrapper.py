"""
Python wrapper for Flash Attention 4 CUDA kernel
"""

import torch
import os
from torch.utils.cpp_extension import load

# Compile CUDA kernel on-the-fly
_flash_attention_4 = None

def _load_cuda_kernel():
    global _flash_attention_4
    if _flash_attention_4 is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cuda_file = os.path.join(current_dir, "flash_attention_4.cu")

        _flash_attention_4 = load(
            name="flash_attention_4",
            sources=[cuda_file],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++17",
                "--use_fast_math",
                "-gencode=arch=compute_90,code=sm_90",  # Blackwell
                "-gencode=arch=compute_89,code=sm_89",  # Ada
                "-gencode=arch=compute_80,code=sm_80",  # Ampere
            ],
            verbose=True
        )
    return _flash_attention_4


class FlashAttention4Function(torch.autograd.Function):
    """
    Flash Attention 4 with automatic differentiation support
    """

    @staticmethod
    def forward(ctx, q, k, v, softmax_scale=None):
        """
        Apply Flash Attention 4

        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_heads, seq_len, head_dim]
            v: Value tensor [batch, num_heads, seq_len, head_dim]
            softmax_scale: Scale factor for softmax (default: 1/sqrt(head_dim))

        Returns:
            output: [batch, num_heads, seq_len, head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Convert to half precision for FA4
        q_half = q.half().contiguous()
        k_half = k.half().contiguous()
        v_half = v.half().contiguous()

        if softmax_scale is None:
            softmax_scale = 1.0 / (head_dim ** 0.5)

        # Allocate output
        output = torch.empty_like(q_half)

        # Load CUDA kernel
        kernel = _load_cuda_kernel()

        # Launch kernel
        stream = torch.cuda.current_stream().cuda_stream
        kernel.launch_flash_attention_4(
            q_half.data_ptr(),
            k_half.data_ptr(),
            v_half.data_ptr(),
            output.data_ptr(),
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            softmax_scale,
            stream
        )

        # Save for backward
        ctx.save_for_backward(q_half, k_half, v_half, output)
        ctx.softmax_scale = softmax_scale

        # Convert back to original dtype
        return output.to(q.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using standard attention
        (FA4 backward kernel would go here in production)
        """
        q, k, v, output = ctx.saved_tensors
        softmax_scale = ctx.softmax_scale

        # For now, use PyTorch's autograd with standard attention
        # In production, this would use a custom backward kernel
        q_fp32 = q.float().requires_grad_(True)
        k_fp32 = k.float().requires_grad_(True)
        v_fp32 = v.float().requires_grad_(True)

        # Recompute attention with autograd
        scores = torch.matmul(q_fp32, k_fp32.transpose(-2, -1)) * softmax_scale
        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v_fp32)

        # Compute gradients
        grad_output_fp32 = grad_output.float()
        out.backward(grad_output_fp32)

        return q_fp32.grad, k_fp32.grad, v_fp32.grad, None


def flash_attention_4(q, k, v, softmax_scale=None):
    """
    Flash Attention 4 - optimized attention for Blackwell GPUs

    Args:
        q: Query [batch, num_heads, seq_len, head_dim]
        k: Key [batch, num_heads, seq_len, head_dim]
        v: Value [batch, num_heads, seq_len, head_dim]
        softmax_scale: Optional scale factor

    Returns:
        Attention output [batch, num_heads, seq_len, head_dim]
    """
    if not torch.cuda.is_available():
        # Fallback to standard attention on CPU
        if softmax_scale is None:
            softmax_scale = 1.0 / (q.shape[-1] ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
        attn_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)

    # Check if FA4 is supported (requires CUDA compute capability 8.0+)
    device_capability = torch.cuda.get_device_capability()
    if device_capability[0] < 8:
        print(f"Warning: Flash Attention 4 requires compute capability 8.0+, "
              f"got {device_capability[0]}.{device_capability[1]}. Falling back to standard attention.")
        if softmax_scale is None:
            softmax_scale = 1.0 / (q.shape[-1] ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
        attn_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)

    return FlashAttention4Function.apply(q, k, v, softmax_scale)


# Test function
def test_flash_attention_4():
    """Test FA4 implementation"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    batch_size = 2
    num_heads = 4
    seq_len = 128
    head_dim = 64

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')

    # FA4 output
    output_fa4 = flash_attention_4(q, k, v)

    # Standard attention output
    softmax_scale = 1.0 / (head_dim ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
    attn_weights = torch.softmax(scores, dim=-1)
    output_ref = torch.matmul(attn_weights, v)

    # Compare
    diff = (output_fa4 - output_ref).abs().max().item()
    print(f"Max difference between FA4 and reference: {diff}")
    print(f"Test {'PASSED' if diff < 1e-2 else 'FAILED'}")


if __name__ == "__main__":
    test_flash_attention_4()
