# Research Ledger

## Current Best Setting

No final best setting is locked. The current direction is to compare naive KV-RVQ against hybrid approaches that use K-RVQ with native V quantization, while tracking attention diagnostics and perceptual video quality side by side.

## Confirmed Findings

- Deep RVQ improves quality but hurts compression ratio.
- Random rotation after normalization did not help much in current trials.

## Failed or Weak Ideas

- Random rotation after normalization appears weak as a standalone improvement.
- Naive KV-RVQ may be limited if V reconstruction error dominates downstream quality loss.

## Open Hypotheses

- V-RVQ error may dominate quality degradation.
- Hybrid K-RVQ plus native V quantization may be better than naive KV-RVQ.
- First/last chunk lossless plus middle RVQ may preserve temporal boundary quality while improving compression.
- Q-aware K optimization may reduce attention error more directly than K reconstruction loss alone.

## Next Experiments

- Measure K and V reconstruction error separately for matched RVQ depths and codebook sizes.
- Compare naive KV-RVQ with hybrid K-RVQ plus native V quantization at matched compression ratios.
- Run first/last chunk lossless plus middle RVQ against all-RVQ and all-native baselines.
- Add attention diagnostics that report QK error and attention KL alongside PSNR, SSIM, and LPIPS.
- Test Q-aware K optimization with fixed prompt lists and unchanged metric accounting.

## Paper Story

Autoregressive video generation is sensitive to KV cache compression because reconstruction error affects both value retrieval and attention routing. A strong paper story should separate K-side routing errors from V-side content errors, then show that hybrid compression can preserve quality better than treating K and V identically.

## Reviewer Risks

- Compression ratio accounting must include all metadata, codebooks, scales, and lossless segments.
- Runtime measurements must separate encode, decode, restore, and generation overheads.
- Attention diagnostics need to correlate with visible quality, not just reconstruction loss.
- Prompt lists, seeds, configs, and codebook hashes must be logged for every reported run.
- Claims about hybrid compression need matched-ratio comparisons against native quantization and RVQ baselines.
