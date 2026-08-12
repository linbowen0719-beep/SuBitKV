# Third-Party Software

The top-level Apache-2.0 license applies to original QuantVideoGen code. Code
under `experiments/` includes modified upstream projects and remains subject to
the upstream terms below.

| Integration | Upstream | Audited base commit | License |
| --- | --- | --- | --- |
| LongCat-Video | <https://github.com/meituan-longcat/LongCat-Video> | `5f60fae4d6ad8f2efdd1b03e58ba6f8d2fab138e` | MIT |
| Self-Forcing | <https://github.com/guandeh17/Self-Forcing> | `33593df3e81fa3ec10239271dd2c100facac6de1` | Apache-2.0 |
| HY-WorldPlay | <https://github.com/Tencent-Hunyuan/HY-WorldPlay> | `ad424f57cf1054521a24b609dc0832afd7cd1a50` | Tencent HY-WorldPlay Community License |

The commits are the upstream snapshots against which the vendored trees were
audited. QuantVideoGen modifies attention, KV-cache storage, quantization,
offload, and launch paths. Consult Git history for the exact patch.

Important: the HY-WorldPlay Community License contains territory, use, notice,
and redistribution restrictions. Anyone sharing or using that integration
must read `experiments/HY-WorldPlay/License.txt` and preserve
`experiments/HY-WorldPlay/NOTICE`.

Model checkpoints and datasets are not redistributed by this repository. They
retain their own upstream licenses and terms.
