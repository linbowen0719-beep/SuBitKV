import torch
import triton
import triton.language as tl

try:
    from triton.language.extra import libdevice
except ImportError:
    # Triton 2.2 exposes the same libdevice-backed functions through
    # ``triton.language.math`` instead of ``triton.language.extra``.
    from triton.language import math as libdevice

def get_configs():
    configs = [triton.Config({"BLOCK_D": 64}, num_stages=1, num_warps=1)]
    for BLOCK_D in [128, 256, 512]:
            for num_warps in [1, 2, 4]:
                for num_stages in [3, 4, 5]:
                    configs.append(
                        triton.Config({'BLOCK_D': BLOCK_D}, num_stages=num_stages, num_warps=num_warps)
                    )
    return configs

@triton.jit
def _quant_pack_kernel(
    X_ptr,
    Y_ptr,
    SY_ptr,
    D: tl.constexpr,
    SCALE_D: tl.constexpr,
    D_AFTER_PACK: tl.constexpr,
    n_bits: tl.constexpr,
    SCALE_IS_E4M3: tl.constexpr,
    Q_BLOCK_SIZE: tl.constexpr, # quantization block size
    PACK_OUTPUT_INT8: tl.constexpr,
    # Autotune
    BLOCK_D: tl.constexpr,
    SCALE_BLOCK_D: tl.constexpr,
    BLOCK_D_AFTER_PACK: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_d = tl.program_id(1)
    
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offset_d < D
    
    if PACK_OUTPUT_INT8:
        offset_d_for_output = pid_d * BLOCK_D_AFTER_PACK + tl.arange(0, BLOCK_D_AFTER_PACK)
        mask_for_output = offset_d_for_output < D_AFTER_PACK
    else:
        offset_d_for_output = offset_d
        mask_for_output = mask
    
    offset_scale_d = pid_d * SCALE_BLOCK_D + tl.arange(0, SCALE_BLOCK_D)
    mask_scale_d = offset_scale_d < SCALE_D

    # Load input
    x_ptr = X_ptr + pid_s * D + offset_d
    x = tl.load(x_ptr, mask=mask, other=0.0)
    

    # Integer Quantization
    x = x.to(tl.float32)
    x = tl.reshape(x, (SCALE_BLOCK_D, Q_BLOCK_SIZE))
    abs_x = tl.abs(x)
    max_x = tl.max(abs_x, axis=1)
    max_x = tl.reshape(max_x, (SCALE_BLOCK_D, 1))

    max_int_value = 2**(n_bits - 1) - 1
    scale_x = max_x / max_int_value
    scale_x = tl.maximum(scale_x, 1e-10)
    
    # If the scaling factor is also being quantized to E4M3
    if SCALE_IS_E4M3:
        scale_x = scale_x.to(tl.float8e4nv)
    
    # Float Quantization
    y = x / scale_x
    y = libdevice.round(y)
    y = tl.maximum(tl.minimum(y, max_int_value), -max_int_value)
    
    # Deal with output shape
    y = tl.reshape(y, (BLOCK_D,))
    y = y.to(tl.int32)
    scale_x = tl.reshape(scale_x, (SCALE_BLOCK_D,))
    scale_x = scale_x.to(SY_ptr.dtype.element_ty)
    
    # Pack output into int8 if needed
    if PACK_OUTPUT_INT8:
        if n_bits == 4:
            y = y + max_int_value
            y = tl.reshape(y, (BLOCK_D // 2, 2))
            shifts = (1 - tl.arange(0, 2)) * 4
            y = tl.sum(y << shifts[None, :], axis=1)
        elif n_bits == 2:
            y = y + max_int_value
            y = tl.reshape(y, (BLOCK_D // 4, 4))
            shifts = (3 - tl.arange(0, 4)) * 2
            y = tl.sum(y << shifts[None, :], axis=1)
            
    y = y.to(Y_ptr.dtype.element_ty)
    
    # tl.device_print("HD After Pack: ", HD_AFTER_PACK)
        
    # Store output
    y_ptr = Y_ptr + pid_s * D_AFTER_PACK + offset_d_for_output
    tl.store(y_ptr, y, mask=mask_for_output)
    
    scale_y_ptr = SY_ptr + pid_s * SCALE_D + offset_scale_d
    tl.store(scale_y_ptr, scale_x, mask=mask_scale_d)
    

def _quant_pack_e4m3_sm89_compat(
    x: torch.Tensor,
    block_size: int,
    num_bits: int,
    pack_output_int8: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the E4M3 scale path without unsupported Triton scalar FP8 casts.

    Triton 2.2 aborts while lowering FP8 conversions for pre-SM90 devices.
    PyTorch supports the same E4M3FN storage conversion on SM89, so this path
    preserves the original operation order: quantize the scale to E4M3FN,
    decode that scale for division, round away from zero, clamp, then pack.
    """

    B, H, S, D = x.shape
    scale_d = D // block_size
    hd = H * D
    hscale_d = H * scale_d
    qmax = 2 ** (num_bits - 1) - 1

    x_flat = x.permute(0, 2, 1, 3).reshape(B * S, hd).contiguous()
    grouped = x_flat.float().reshape(B * S, hscale_d, block_size)
    scale_float = (grouped.abs().amax(dim=-1, keepdim=True) / qmax).clamp_min(
        1e-10
    )
    scales = scale_float.to(torch.float8_e4m3fn)
    normalized = grouped / scales.float()
    quantized = torch.sign(normalized) * torch.floor(normalized.abs() + 0.5)
    quantized = quantized.clamp(-qmax, qmax).to(torch.int32)
    quantized = quantized.reshape(B * S, hd)

    if pack_output_int8:
        pack_factor = 8 // num_bits
        shifts = torch.arange(
            8 - num_bits,
            -1,
            -num_bits,
            device=x.device,
            dtype=torch.int32,
        )
        packed = (quantized + qmax).reshape(B * S, hd // pack_factor, pack_factor)
        y = torch.sum(packed << shifts, dim=-1).to(torch.uint8)
        y = y.reshape(B, S, H, D // pack_factor).permute(0, 2, 1, 3)
    else:
        y = quantized.to(torch.int8).reshape(B, S, H, D).permute(0, 2, 1, 3)

    scales = scales.reshape(B, S, H, scale_d).permute(0, 2, 1, 3)
    return y, scales


def quant_pack(
    x: torch.Tensor,
    block_size: int,
    num_bits: int,
    scale_precision: torch.dtype,
    pack_output_int8: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor with block-wise scaling.

    Performs block-wise quantization on the input tensor. Each block of size
    `block_size` along the last dimension is quantized with a shared scale factor.

    Args:
        x: Tensor of shape (B, H, S, D), the input tensor to quantize.
           D should be divisible by block_size.
        block_size: The size of each block for quantization.
        num_bits: The number of bits for quantization.
        scale_precision: The dtype for storing scale factors. Must be bfloat16 or float8_e4m3fn.
        pack_output_int8: If True, pack the output into int8. In this case, num_bits must be 2 or 4.
        
    Returns:
        x_quant: Quantized tensor.
        scales: Scale factors with dtype `scale_precision`, shape (..., D // block_size).
    """
    assert num_bits in (2, 3, 4, 8), "num_bits must be 2, 3, 4, or 8"
    assert block_size > 0 and block_size & (block_size - 1) == 0, "block_size must be a power of two"
    assert scale_precision in (torch.bfloat16, torch.float8_e4m3fn), "scale_precision must be bfloat16 or float8_e4m3fn"
    if pack_output_int8:
        assert num_bits in (2, 4), "num_bits must be 2 or 4 when pack_output_int8 is True"
    
    B, H, S, D = x.shape
    assert D % block_size == 0, "last dimension must be divisible by block_size"

    if (
        scale_precision == torch.float8_e4m3fn
        and x.is_cuda
        and torch.cuda.get_device_capability(x.device) < (9, 0)
    ):
        return _quant_pack_e4m3_sm89_compat(
            x,
            block_size=block_size,
            num_bits=num_bits,
            pack_output_int8=pack_output_int8,
        )

    SCALE_D = D // block_size
    
    HD = H * D
    HSCALE_D = H * SCALE_D
    
    SCALE_IS_E4M3 = scale_precision == torch.float8_e4m3fn
    PACK_OUTPUT_INT8 = pack_output_int8 and num_bits in (2, 4)
    
    # First, reshape to (B * S, H * D) while keeping the original shape
    x = x.permute(0, 2, 1, 3).reshape(B * S, H * D).contiguous()
    
    # Define Outputs
    if PACK_OUTPUT_INT8:
        elem_per_int = 8 // num_bits
        HD_AFTER_PACK = HD // elem_per_int
        y = torch.zeros(B * S, HD // elem_per_int, device=x.device, dtype=torch.uint8)
    else:
        y = torch.zeros(B * S, HD, device=x.device, dtype=torch.int8)
        HD_AFTER_PACK = HD
    scales = torch.zeros(B * S, HSCALE_D, device=x.device, dtype=scale_precision)

    block_d = 128
    if HD < block_d:
        block_d = triton.next_power_of_2(HD)
    block_d = max(block_size, block_d)
    if block_d % block_size != 0:
        block_d = triton.next_power_of_2(block_d)
    scale_block_d = block_d // block_size
    block_d_after_pack = block_d // (8 // num_bits) if PACK_OUTPUT_INT8 else block_d

    grid = (B * S, triton.cdiv(HD, block_d))
    _quant_pack_kernel[grid](
        x,
        y,
        scales,
        HD,
        HSCALE_D,
        HD_AFTER_PACK,
        num_bits,
        SCALE_IS_E4M3,
        block_size,
        PACK_OUTPUT_INT8,
        BLOCK_D=block_d,
        SCALE_BLOCK_D=scale_block_d,
        BLOCK_D_AFTER_PACK=block_d_after_pack,
        num_warps=4,
    )
    
    # Reshape to original shape
    if PACK_OUTPUT_INT8:
        y = y.reshape(B, S, H, D // elem_per_int).permute(0, 2, 1, 3)
    else:
        y = y.reshape(B, S, H, D).permute(0, 2, 1, 3)
    scales = scales.reshape(B, S, H, SCALE_D).permute(0, 2, 1, 3)
    
    return y, scales


if __name__ == "__main__":
    x = torch.randn(1, 4, 1, 128, device="cuda").to(torch.bfloat16)
    y, scales = quant_pack(x, block_size=16, num_bits=4, scale_precision=torch.float8_e4m3fn, pack_output_int8=True)
    print(y)
    print(scales)
