from dataclasses import dataclass
import re


_LAYER_TOKEN_RE = re.compile(r"^(?P<start>\d+)(?:-(?P<end>\d+))?$")
_RESIDUAL_CODEC_SPAN_ALIASES = {
    "q": "quantvideo",
    "qv": "quantvideo",
    "quantvideo": "quantvideo",
    "s": "structured_truncatable",
    "structured": "structured_truncatable",
    "structured_truncatable": "structured_truncatable",
    "r": "r2aq",
    "r2aq": "r2aq",
}


def parse_layer_spec(value: str) -> tuple[int, ...]:
    """Parse a deterministic comma/range layer specification.

    ``"0,4-7,10"`` becomes ``(0, 4, 5, 6, 7, 10)``. Empty strings select no
    layers. Malformed, negative, descending, or empty range tokens fail closed.
    """

    spec = str(value or "").strip()
    if not spec:
        return ()
    layers: set[int] = set()
    for raw_token in spec.split(","):
        token = raw_token.strip()
        match = _LAYER_TOKEN_RE.fullmatch(token)
        if match is None:
            raise ValueError(
                f"invalid layer token {raw_token!r} in {value!r}; expected N or N-M"
            )
        start = int(match.group("start"))
        end_text = match.group("end")
        end = start if end_text is None else int(end_text)
        if end < start:
            raise ValueError(
                f"descending layer range {token!r} in {value!r} is not allowed"
            )
        layers.update(range(start, end + 1))
    return tuple(sorted(layers))


def parse_residual_codec_span_schedule(value: str) -> tuple[str, ...]:
    """Parse a fixed physical-span residual-codec schedule.

    The compact spelling ``"q,q,q,s,s"`` is canonicalized to three
    ``quantvideo`` spans followed by two ``structured_truncatable`` spans.
    The schedule is a shared decoder configuration, not per-token metadata.
    Empty strings disable span routing. Unknown or empty entries fail closed.
    """

    spec = str(value or "").strip().lower()
    if not spec:
        return ()
    codecs: list[str] = []
    for raw_token in spec.split(","):
        token = raw_token.strip()
        if not token:
            raise ValueError(
                f"empty residual-codec span token in {value!r}"
            )
        canonical = _RESIDUAL_CODEC_SPAN_ALIASES.get(token)
        if canonical is None:
            raise ValueError(
                f"unsupported residual-codec span token {raw_token!r}; "
                "expected q/quantvideo, s/structured_truncatable, or r/r2aq"
            )
        codecs.append(canonical)
    return tuple(codecs)


@dataclass
class QuantizeConfig:
    """Configuration for model quantization settings."""

    quant_type: str = "none"
    """Quantization type: 'none', 'naive-fp4', 'kmeans-fp4'."""

    # KV cache quantization parameters
    cache_num_k_centroids: int = 256
    """Number of K-Means centroids for K tensor (used in kmeans and nstages-kmeans)."""

    cache_num_v_centroids: int = 256
    """Number of K-Means centroids for V tensor (used in kmeans and nstages-kmeans)."""

    kmeans_max_iters: int = 100
    """Maximum iterations for K-Means clustering."""

    quant_block_size: int = 16
    """Block size for quantization."""

    kivi_group_size: int = 32
    """Token/channel group size for the all-compressed KIVI-style adapter."""

    turboquant_seed: int = 20260803
    """Local seed for fixed TurboQuant rotation and QJL matrices."""

    # PRQ (nstages-kmeans) specific parameters
    num_prq_stages: int = 4
    """Number of PRQ stages for nstages-kmeans quantization."""

    rvq_residual_quantize: bool = False
    """Quantize the BF16 RVQ reconstruction residual with the selected QuantVideo codec."""

    temporal_adaptive_residual: bool = False
    """Admit each adaptive-RVQ chunk at B16 and only vary residual precision."""

    temporal_adaptive_profile_path: str = ""
    """Frozen train-only time/rate-distortion profile for residual scheduling."""

    temporal_adaptive_capacity_bit_chunks: int = 40
    """Peak residual bit count; base-aware profiles impose per-step byte-derived limits."""

    temporal_adaptive_allocation_mode: str = "optimal"
    """Residual scheduler: optimal, or force_recent_b16 counterfactual."""

    temporal_adaptive_fast_identity: bool = True
    frame_mean_spatial_tokens: int = 880
    calibration_sampling_mode: str = "random"
    """Use synchronization-free tensor-version guards during admission."""

    rvq_residual_codec: str = "quantvideo"
    """RVQ residual codec, including uniform or static allocated structured prefixes."""

    rvq_k_residual_codec: str = ""
    """Optional K-only RVQ residual codec; empty inherits rvq_residual_codec."""

    rvq_v_residual_codec: str = ""
    """Optional V-only RVQ residual codec; empty inherits rvq_residual_codec."""

    kivi_residual_bits: int = 4
    """Bits used by KIVI-style affine quantization of an RVQ residual."""

    kivi_residual_group_size: int = 32
    """Token/channel group size for K/V KIVI-style RVQ residuals."""

    rvq_k_structured_layers: str = ""
    """K layers using structured_truncatable; other K layers use quantvideo."""

    rvq_residual_codec_span_schedule: str = ""
    """Fixed comma-separated codec choice for each physical history span."""

    higgs_group_size: int = 32
    """HIGGS-lite channel block size (32 or 64)."""

    higgs_refine_iters: int = 1
    """HIGGS-lite closed-form scale refinement iterations (0 or 1)."""

    higgs_seed: int = 20260722
    """Global deterministic HIGGS-lite random-sign seed."""

    truncatable_residual_scheme: str = "nested_affine"
    """Nested residual codec family used when rvq_residual_codec=truncatable."""

    truncatable_residual_bits: int = 4
    """Number of retained prefix bitplanes per residual value (1--4)."""

    truncatable_residual_params_path: str = ""
    """Shared K/V tree, alpha, and vector-codebook parameter file."""

    truncatable_residual_hadamard: bool = False
    """Apply the selected codec's fixed groupwise sign-Hadamard transform."""

    truncatable_residual_seed: int = 20260722
    """Deterministic random-sign seed for optional Hadamard preprocessing."""

    routed_gaussian_bits: int = 4
    """Nominal routed Gaussian prefix depth; B1 is uniform by definition."""

    routed_gaussian_group_size: int = 128
    """FP8 RMS scale group for the cluster-free routed Gaussian codec."""

    routed_gaussian_hadamard_group_size: int = 128
    """Fixed sign-Hadamard channel group for routed Gaussian residuals."""

    routed_gaussian_scale_refine_iters: int = 1
    """Closed-form FP8 scale-refinement passes for routed Gaussian residuals."""

    routed_gaussian_seed: int = 20260722
    """Global fixed sign-Hadamard seed for routed Gaussian residuals."""

    routed_gaussian_routing_path: str = ""
    """Static layer/head K/V B-1/B/B+1 route table."""

    routed_gaussian_tree_params_path: str = ""
    """Optional fixed K/V Gaussian tree parameter file."""

    selective_rank1_gaussian_params_path: str = ""
    """Train-only sparse K rank-1 basis used before the Gaussian suffix."""

    absmax_scalar_bits: int = 4
    """Signed scalar bit width for cluster-free block-absmax RVQ residuals."""

    absmax_scalar_group_size: int = 128
    """Channel block size sharing one FP8 absmax scale."""

    sign_adaptive_absmax_group_size: int = 128
    """Channel block sharing one signed FP8 scale for sign-adaptive INT4."""

    progressive_v_absmax_bits: int = 4
    """Retained prefix depth for the B4-preserving progressive V codec."""

    progressive_v_absmax_group_size: int = 128
    """Channel group sharing the unchanged signed FP8 absmax scale."""

    progressive_v_absmax_method: str = "midpoint_msb"
    """Train-only incomplete-prefix reconstruction table to use."""

    progressive_v_absmax_params_path: str = ""
    """Shared prefix bit-order and B1/B2/B3 reconstruction parameter file."""

    centered_absmax_scalar_bits: int = 4
    """Signed scalar bit width after a cluster-free per-token offset."""

    centered_absmax_scalar_group_size: int = 128
    """Channel group sharing one FP8 offset and one FP8 absmax scale."""

    centered_absmax_scalar_center_mode: str = "mean"
    """Per-token group offset: arithmetic mean or min/max midpoint."""

    routed_midrange_scalar_bits: int = 4
    """Nominal static-routed prefix depth for midrange scalar residuals."""

    static_equalized_midrange_params_path: str = ""
    """Train-only FP8 K channel equalization table; V remains uniform."""

    structured_residual_group_size: int = 128
    """Channel group size for the nested suffix after online vector prediction."""

    structured_residual_clusters: int = 256
    """Online per-head vector-predictor centroids for structured truncation."""

    structured_residual_kmeans_iters: int = 2
    """Online vector-predictor K-Means iterations."""

    structured_residual_kmeans_seed: int = 42
    """Deterministic K-Means initialization seed for structured truncation."""

    structured_residual_scale_refine_iters: int = 1
    """Closed-form shared-scale refinement passes for all suffix prefixes."""

    structured_residual_k_bits: int = 0
    """Optional K-only prefix depth; zero inherits truncatable_residual_bits."""

    structured_residual_v_bits: int = 0
    """Optional V-only prefix depth; zero inherits truncatable_residual_bits."""

    structured_residual_fast_decode: bool = False
    """Opt into the fail-closed fused G128 Gaussian/Hadamard CUDA decoder."""

    structured_prefix_allocation_path: str = ""
    """Shared layer/head K/V prefix table for structured_static_prefix."""

    r2aq_residual_bits: int = 4
    """Nominal R²AQ suffix depth; the recycled budget adds a sparse next plane."""

    r2aq_k_bits: int = 0
    """Optional K-only R²AQ nominal depth; zero inherits r2aq_residual_bits."""

    r2aq_v_bits: int = 0
    """Optional V-only R²AQ nominal depth; zero inherits r2aq_residual_bits."""

    r2aq_predictor_group_size: int = 128
    """Channel group used to store online C256 predictor centroids as INT4."""

    r2aq_suffix_group_size: int = 64
    """Channel group used by the QV-anchored embedded residual tree."""

    r2aq_clusters: int = 256
    """Number of online vector-predictor centroids used by R²AQ."""

    r2aq_kmeans_iters: int = 2
    """Online vector-predictor K-Means iterations used by R²AQ."""

    r2aq_kmeans_seed: int = 42
    """Deterministic online vector-predictor seed used by R²AQ."""

    r2aq_selection_mode: str = "scale"
    """Sparse refinement selector: scale or scale_utility."""

    r2aq_ablation: str = ""
    """Explicit R²AQ ablation; empty preserves legacy selection-mode routing."""

    r2aq_selection_fraction: float | None = None
    """Optional fraction of the reclaimed predictor budget spent on refinement."""

    r2aq_coherent_tail_group_fraction: float = 0.55
    """Fraction of candidate coherent-tail groups requested by the encoder."""

    r2aq_coherent_tail_group_size: int = 128
    """Channel width of coherent-tail corrections (G128 for the staged codec)."""

    r2aq_utility_weights_path: str = ""
    """Optional shared layer/KV/age/channel utility table for R²AQ."""

    r2aq_tree_params_path: str = ""
    """Optional shared K/V learned nested-tree parameters for R²AQ."""

    r2aq_fast_decode: bool = False
    """Opt into the fail-closed fused CUDA R²AQ decoder."""

    r2aq_coherent_tail_eager_decode: bool = False
    """Explicitly allow eager decode only for coherent-tail R²AQ V states."""

    r2aq_shared_budget_owner_layer: int = -1
    """Layer of the unique state that repays globally shared R²AQ tables."""

    r2aq_shared_budget_owner_span: int = -1
    """Physical-span index of the unique shared-budget owner state."""

    r2aq_payload_logical_multiplier: int = 1
    """Runtime accounting multiplier from one core state to logical KV bytes."""

    # Runtime RVQ progressive-depth parameters
    rvq_progressive: bool = False
    """Use adaptive prefix depth for runtime RVQ encoding."""

    rvq_progressive_min_depth: int = 0
    """Minimum RVQ depth used for every vector when progressive RVQ is enabled."""

    rvq_progressive_max_depth: int = 0
    """Maximum RVQ depth used for any vector when progressive RVQ is enabled."""

    rvq_progressive_error_threshold: float = 0.0
    """Normalized residual RMS threshold for stopping progressive RVQ encoding."""

    rvq_progressive_granularity: str = "token"
    """Progressive RVQ depth granularity: token, layer, or block."""

    rvq_progressive_block_size: int = 880
    """Token block size for block-level progressive RVQ."""

    rvq_relative_stop: bool = False
    """Stop dense span-level RVQ after consecutive weak MSE reductions."""

    rvq_relative_stop_threshold: float = 0.0629
    """Per-stage relative residual-MSE reduction required to continue RVQ."""

    rvq_relative_stop_min_depth: int = 4
    """Minimum dense RVQ prefix retained by relative stopping."""

    rvq_relative_stop_max_depth: int = 16
    """Maximum dense RVQ prefix considered by relative stopping."""

    rvq_relative_stop_fast_approximate: bool = False
    """Use one monolithic kernel; only FP32 near-tie assignments may differ."""

    rvq_relative_stop_monolithic_exact: bool = False
    """Use monolithic assignment plus exact BF16 early-stop MSE replay."""

    rvq_relative_stop_exact_staged_k_layers: tuple[int, ...] = ()
    """K layers that retain staged assignment under monolithic-exact mode."""

    rvq_relative_stop_exact_staged_v_layers: tuple[int, ...] = ()
    """V layers that retain staged assignment under monolithic-exact mode."""

    rvq_k_calibration: str = "none"
    """Post-RVQ K calibration mode: none, online, or offline."""

    rvq_v_calibration: str = "none"
    """Post-RVQ V calibration mode: none, online, or offline."""

    rvq_calibration_path: str = ""
    """Offline post-RVQ calibration table path."""

    rvq_calibration_block_size: int = 880
    """Token block size for coarse post-RVQ calibration scales."""

    rvq_calibration_min: float = 0.75
    """Lower clamp for post-RVQ calibration scales."""

    rvq_calibration_max: float = 1.33
    """Upper clamp for post-RVQ calibration scales."""

    rvq_calibration_storage_dtype: str = "uint4"
    """Runtime storage dtype for post-RVQ calibration scales."""
