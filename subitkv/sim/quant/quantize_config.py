from dataclasses import dataclass


@dataclass
class QuantizeConfig:
    """Internal settings for persistent bases and adaptive K/V residuals."""

    quant_type: str = "none"
    cache_num_k_centroids: int = 256
    cache_num_v_centroids: int = 256
    kmeans_max_iters: int = 100
    quant_block_size: int = 16
    num_prq_stages: int = 4
    k_rvq: bool = False
    v_rvq: bool = False
    kivi_group_size: int = 32
    turboquant_seed: int = 20260803
    rvq_residual_quantize: bool = False
    temporal_adaptive_residual: bool = False
    temporal_adaptive_profile_path: str = ""
    temporal_adaptive_capacity_bit_chunks: int = 40
    temporal_adaptive_allocation_mode: str = "optimal"
    temporal_adaptive_fast_identity: bool = True
    frame_mean_spatial_tokens: int = 880
    calibration_sampling_mode: str = "random"
    rvq_k_residual_codec: str = ""
    rvq_v_residual_codec: str = ""
    routed_gaussian_bits: int = 4
    routed_gaussian_group_size: int = 128
    routed_gaussian_hadamard_group_size: int = 128
    routed_gaussian_scale_refine_iters: int = 1
    routed_gaussian_seed: int = 20260722
    routed_gaussian_routing_path: str = ""
    routed_gaussian_tree_params_path: str = ""
    progressive_v_absmax_bits: int = 4
    progressive_v_absmax_group_size: int = 128
    progressive_v_absmax_method: str = "midpoint_msb"
    progressive_v_absmax_params_path: str = ""
    rvq_relative_stop: bool = False
    rvq_relative_stop_threshold: float = 0.0629
    rvq_relative_stop_min_depth: int = 4
    rvq_relative_stop_max_depth: int = 16
    rvq_relative_stop_fast_approximate: bool = False
    rvq_relative_stop_monolithic_exact: bool = False
    rvq_relative_stop_exact_staged_k_layers: tuple[int, ...] = ()
    rvq_relative_stop_exact_staged_v_layers: tuple[int, ...] = ()
