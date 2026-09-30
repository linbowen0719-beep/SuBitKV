import math

import torch


CODEC_NAME = "block_fixed_length_v1"


def unpack_lowbit_tensor(packed: torch.Tensor, num_bits: int, original_dim: int) -> torch.Tensor:
    """Unpack QuantVideo's fixed-width uint8 residual format to unsigned integers."""
    if num_bits == 4:
        high = (packed >> 4) & 0xF
        low = packed & 0xF
        values = torch.stack((high, low), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    elif num_bits == 2:
        v1 = (packed >> 6) & 0x3
        v2 = (packed >> 4) & 0x3
        v3 = (packed >> 2) & 0x3
        v4 = packed & 0x3
        values = torch.stack((v1, v2, v3, v4), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 4)
    else:
        raise ValueError(f"block fixed-length codec only supports int2/int4, got int{num_bits}")

    return values[..., :original_dim].contiguous()


def pack_lowbit_tensor(values: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Pack unsigned low-bit integers into QuantVideo's fixed-width uint8 format."""
    if num_bits == 4:
        if values.shape[-1] % 2 != 0:
            values = torch.nn.functional.pad(values, (0, 1))
        high = values[..., 0::2]
        low = values[..., 1::2]
        return ((high << 4) | low).to(torch.uint8).contiguous()

    if num_bits == 2:
        pad_dim = (-values.shape[-1]) % 4
        if pad_dim:
            values = torch.nn.functional.pad(values, (0, pad_dim))
        v1 = values[..., 0::4]
        v2 = values[..., 1::4]
        v3 = values[..., 2::4]
        v4 = values[..., 3::4]
        return ((v1 << 6) | (v2 << 4) | (v3 << 2) | v4).to(torch.uint8).contiguous()

    raise ValueError(f"block fixed-length codec only supports int2/int4, got int{num_bits}")


def _metadata_pack(mins: torch.Tensor, bits: torch.Tensor, num_bits: int) -> torch.Tensor:
    if num_bits == 4:
        return ((mins << 4) | bits).to(torch.uint8).contiguous()
    if num_bits == 2:
        return ((mins << 2) | bits).to(torch.uint8).contiguous()
    raise ValueError(f"block fixed-length codec only supports int2/int4, got int{num_bits}")


def _metadata_unpack(metadata: torch.Tensor, num_bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    if num_bits == 4:
        mins = (metadata >> 4) & 0xF
        bits = metadata & 0xF
    elif num_bits == 2:
        mins = (metadata >> 2) & 0x3
        bits = metadata & 0x3
    else:
        raise ValueError(f"block fixed-length codec only supports int2/int4, got int{num_bits}")
    return mins.contiguous(), bits.contiguous()


def _bit_offsets(bits_flat: torch.Tensor, block_size: int) -> tuple[torch.Tensor, int]:
    bit_lengths = bits_flat.to(torch.int64) * block_size
    offsets = torch.cumsum(bit_lengths, dim=0) - bit_lengths
    total_bits = int(bit_lengths.sum().item())
    return offsets, total_bits


def _pack_variable_width_offsets(
    offsets_flat: torch.Tensor,
    bits_flat: torch.Tensor,
    block_size: int,
    num_bits: int,
) -> tuple[torch.Tensor, int]:
    bit_offsets, total_bits = _bit_offsets(bits_flat, block_size)
    payload = torch.zeros(
        (math.ceil(total_bits / 8),),
        device=offsets_flat.device,
        dtype=torch.int32,
    )

    block_positions = torch.arange(block_size, device=offsets_flat.device, dtype=torch.int64)
    for width in range(1, num_bits + 1):
        mask = bits_flat == width
        if not bool(mask.any().item()):
            continue

        starts = bit_offsets[mask]
        values = offsets_flat[mask].to(torch.int64)
        element_bit_offsets = block_positions.view(1, -1) * width
        for bit in range(width):
            bit_positions = starts.view(-1, 1) + element_bit_offsets + bit
            byte_indices = (bit_positions >> 3).reshape(-1)
            byte_shifts = (bit_positions & 7).reshape(-1).to(torch.int32)
            bit_values = ((values >> bit) & 1).reshape(-1).to(torch.int32) << byte_shifts
            payload.index_put_((byte_indices,), bit_values, accumulate=True)

    return payload.to(torch.uint8), total_bits


def _unpack_variable_width_offsets(
    payload: torch.Tensor,
    bits_flat: torch.Tensor,
    block_size: int,
    num_bits: int,
) -> torch.Tensor:
    bit_offsets, _ = _bit_offsets(bits_flat, block_size)
    offsets_flat = torch.empty(
        (bits_flat.numel(), block_size),
        device=payload.device,
        dtype=torch.uint8,
    )

    block_positions = torch.arange(block_size, device=payload.device, dtype=torch.int64)
    for width in range(1, num_bits + 1):
        mask = bits_flat == width
        if not bool(mask.any().item()):
            continue

        starts = bit_offsets[mask]
        element_bit_offsets = block_positions.view(1, -1) * width
        values = torch.zeros((starts.numel(), block_size), device=payload.device, dtype=torch.uint8)
        for bit in range(width):
            bit_positions = starts.view(-1, 1) + element_bit_offsets + bit
            byte_indices = bit_positions >> 3
            byte_shifts = bit_positions & 7
            bit_values = ((payload[byte_indices] >> byte_shifts) & 1).to(torch.uint8)
            values |= bit_values << bit

        offsets_flat[mask] = values

    return offsets_flat


def encode_block_fixed_length_residual(
    residual_quant: torch.Tensor,
    num_bits: int,
    original_dim: int,
    block_size: int = 16,
) -> dict:
    """
    Losslessly encode QuantVideo's packed INT2/INT4 residual with Vayne-style
    block-wise fixed-length groups along the sequence dimension.

    This is an experimental codec. It stores real packed payload bytes and one
    metadata byte per (B, H, sequence_block, D) group. The one-byte metadata is
    slightly less compact than Vayne's theoretical bit estimate, but keeps the
    implementation simple and decodable by the current Python/Torch path.
    """
    if num_bits not in (2, 4):
        raise ValueError(f"block fixed-length codec only supports int2/int4, got int{num_bits}")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")

    residual_quant = residual_quant.contiguous()
    bsz, heads, seq_len, _ = residual_quant.shape
    values = unpack_lowbit_tensor(residual_quant, num_bits, original_dim)

    padded_seq_len = math.ceil(seq_len / block_size) * block_size
    if padded_seq_len != seq_len:
        values = torch.nn.functional.pad(values, (0, 0, 0, padded_seq_len - seq_len))

    num_blocks = padded_seq_len // block_size
    groups = values.reshape(bsz, heads, num_blocks, block_size, original_dim)
    groups = groups.permute(0, 1, 2, 4, 3).contiguous()

    mins = groups.min(dim=-1).values
    maxs = groups.max(dim=-1).values
    diffs = (maxs - mins).clamp(min=1)
    bits = torch.ceil(torch.log2(diffs.float() + 1)).to(torch.uint8)
    metadata = _metadata_pack(mins, bits, num_bits)

    offsets_flat = (groups - mins.unsqueeze(-1)).reshape(-1, block_size).to(torch.uint8)
    bits_flat = bits.reshape(-1).contiguous()
    payload, payload_num_bits = _pack_variable_width_offsets(
        offsets_flat,
        bits_flat,
        block_size,
        num_bits,
    )

    vayne_metadata_bits = num_bits + math.ceil(math.log2(num_bits + 1))
    vayne_estimated_bits = int((bits_flat.to(torch.int64) * block_size + vayne_metadata_bits).sum().item())

    return {
        "codec": CODEC_NAME,
        "num_bits": num_bits,
        "block_size": block_size,
        "original_shape": tuple(residual_quant.shape),
        "original_dim": original_dim,
        "seq_len": seq_len,
        "padded_seq_len": padded_seq_len,
        "payload_num_bits": payload_num_bits,
        "vayne_estimated_bits": vayne_estimated_bits,
        "metadata": metadata,
        "payload": payload,
    }


def decode_block_fixed_length_residual(encoded: dict) -> torch.Tensor:
    """Decode a block fixed-length residual back to QuantVideo's packed uint8 format."""
    if encoded.get("codec") != CODEC_NAME:
        raise ValueError(f"Unsupported residual codec: {encoded.get('codec')}")

    num_bits = int(encoded["num_bits"])
    block_size = int(encoded["block_size"])
    original_shape = tuple(encoded["original_shape"])
    original_dim = int(encoded["original_dim"])
    seq_len = int(encoded["seq_len"])
    padded_seq_len = int(encoded["padded_seq_len"])
    metadata = encoded["metadata"].contiguous()
    payload = encoded["payload"].contiguous()

    mins, bits = _metadata_unpack(metadata, num_bits)
    bits_flat = bits.reshape(-1).contiguous()
    offsets_flat = _unpack_variable_width_offsets(payload, bits_flat, block_size, num_bits)

    values = (offsets_flat + mins.reshape(-1, 1)).reshape(*metadata.shape, block_size)
    bsz, heads, num_blocks, dim, _ = values.shape
    values = values.reshape(bsz, heads, num_blocks, dim, block_size)
    values = values.permute(0, 1, 2, 4, 3).reshape(bsz, heads, padded_seq_len, dim)
    values = values[:, :, :seq_len, :original_dim].contiguous()

    residual_quant = pack_lowbit_tensor(values, num_bits)
    return residual_quant.reshape(original_shape).contiguous()


def is_block_fixed_length_residual(value) -> bool:
    return isinstance(value, dict) and value.get("codec") == CODEC_NAME
