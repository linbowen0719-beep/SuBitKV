"""Dispatch the historical video-cache baselines without pretrained assets."""

import re

BASELINE_QUANT_TYPES = {
    "kivi_int2": "kivi-all-int2",
    "kivi_int4": "kivi-all-int4",
    "turboquant_int2": "turboquant-prod-k2-v2",
    "turboquant_int4": "turboquant-prod-k4-v4",
    "quantvideo_int2": "triton-nstages-kmeans-int2",
    "quantvideo_int4": "triton-nstages-kmeans-int4",
}


def is_baseline(quant_type, config):
    return bool(
        re.fullmatch(r"kivi-all-int(2|4)", quant_type)
        or re.fullmatch(r"turboquant-(mse|prod)-k(2|3|4)-v(2|3|4)", quant_type)
        or (
            quant_type in {"triton-nstages-kmeans-int2", "triton-nstages-kmeans-int4"}
            and not getattr(config, "k_rvq", False)
            and not getattr(config, "v_rvq", False)
        )
    )


def encode_pair(k, v, quant_type, config):
    import torch

    from .rng import preserve_torch_rng_state

    kivi = re.fullmatch(r"kivi-all-int(2|4)", quant_type)
    turbo = re.fullmatch(r"turboquant-(mse|prod)-k(2|3|4)-v(2|3|4)", quant_type)
    if kivi:
        from .kivi import encode_kivi_tensor

        states = tuple(
            encode_kivi_tensor(
                x, tensor_name=name, bits=int(kivi[1]),
                group_size=getattr(config, "kivi_group_size", 32),
            )
            for name, x in (("k", k), ("v", v))
        )
    elif turbo:
        from .turboquant import encode_turboquant_mse, encode_turboquant_prod

        pair = (int(turbo[2]), int(turbo[3]))
        if pair not in {(2, 2), (3, 2), (4, 3), (4, 4)}:
            raise ValueError(f"unsupported TurboQuant K/V bit pair: {pair}")
        seed = getattr(config, "turboquant_seed", 20260803)
        encode_k = encode_turboquant_prod if turbo[1] == "prod" else encode_turboquant_mse
        states = (
            encode_k(k, bits=pair[0], seed=seed),
            encode_turboquant_mse(v, bits=pair[1], seed=seed),
        )
    else:
        if not k.is_cuda or not v.is_cuda:
            raise ValueError("QuantVideo's historical Triton implementation requires CUDA")
        from .functions import triton_prq_quantize_tensor

        bits = int(quant_type[-1])
        states = []
        # Online K-means must not change the generator's future noise draws.
        with preserve_torch_rng_state(k.device):
            for name, x in (("k", k), ("v", v)):
                state = triton_prq_quantize_tensor(
                    x, num_stages=config.num_prq_stages,
                    num_clusters=getattr(config, f"cache_num_{name}_centroids"),
                    block_size=config.quant_block_size, max_iters=config.kmeans_max_iters,
                    scale_precision=torch.float8_e4m3fn, quantize_fn=lambda _: bits,
                )
                state.update(quant_method="quantvideo", bits=bits, block_size=config.quant_block_size)
                states.append(state)
    for state, x in zip(states, (k, v)):
        state["info"] = {"output_dtype": x.dtype}
    return tuple(states)


def decode_state(state, output_dtype):
    method = state.get("quant_method")
    if method == "kivi_all_compressed":
        from .kivi import decode_kivi_tensor

        return decode_kivi_tensor(state, output_dtype)
    if method in {"turboquant_mse", "turboquant_prod"}:
        from .turboquant import decode_turboquant

        return decode_turboquant(state, output_dtype)
    if method == "quantvideo":
        from .functions import triton_prq_dequantize_tensor

        return triton_prq_dequantize_tensor(
            state, state["block_size"], state["bits"], output_dtype=output_dtype
        )
    raise ValueError(f"unsupported baseline representation: {method}")
