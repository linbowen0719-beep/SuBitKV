# Research Index

Only `configs/smoke/`, `configs/main/`, and the commands in
`docs/reproduction.md` are supported reproduction interfaces. The remaining
research material is retained to make decisions and evidence traceable.

## Current synthesis

- `wan2_2_5b_adaptive_rvq_progressive_kv_compression_paper_zh.md`: cross-model
  Adaptive RVQ and progressive KV-compression report.
- `wan2_2_5b_kv_compression_paper_story_and_analysis_zh.md`: current paper
  narrative and experiment matrix.
- `wan2_2_5b_r2aq_results_zh.md`: R²AQ evidence integration.
- `research_ledger.md`: confirmed findings, open hypotheses, and reviewer risks.

## Method and benchmark notes

- `rvq_kv_storage.md`: RVQ storage accounting.
- `progressive_rvq_weakness_analysis.md`: progressive RVQ limitations.
- `qk_scale_weakness_analysis.md`: QK-scale diagnostics.
- Files named `wan2_2_5b_*_zh.md`: dated experiment reports and protocol notes.

Many reports intentionally preserve absolute evidence paths and hashes from the
machine on which they were produced. These paths are provenance records; they
must not be copied into new supported configs. When promoting a historical
experiment, create a portable config, document data preparation, and add a
dry-run test first.
