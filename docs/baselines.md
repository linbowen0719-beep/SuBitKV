# Video KV baselines

The historical KIVI-style and TurboQuant modules, direct bitpacking, and
QuantVideo's K-means, PRQ, residual packing and decoding dependencies are
restored from the parent of release commit `9046baf`. The current dispatch and
native model adapters accept all six INT2/INT4 baseline methods.

| Method | Implementation | Setting |
| --- | --- | --- |
| KIVI | `subitkv/kivi.py` | Asymmetric K per-channel/token-group and V per-token/channel-group affine quantization, group size 32 |
| TurboQuant | `subitkv/turboquant.py` | D128 sphere Lloyd-Max tables and seeded orthogonal rotation; K uses product correction, V uses MSE quantization |
| QuantVideo | `subitkv/functions.py`, `subitkv/real/`, `subitkv/sim/quant/lowbit_quantize.py` | Online K-means PRQ followed by packed INT2/INT4 residuals |

KIVI compresses every completed span; it does not retain the original method's
recent-token BF16 window. These are the repository's historical video adapters,
not a claim that all upstream default settings or published tables are reproduced.
TurboQuant supports D128 and requires SciPy. QuantVideo requires CUDA/Triton;
the restored launch preset uses one PRQ stage, 256 centroids, two K-means
iterations and residual block size 64. Its randomized online fitting preserves
the model's random stream.

After checkpoint download, inference runs BF16 plus the selected baseline:

```bash
python scripts/run_subitkv.py infer --model wan2_2_5b --run-id demo --baseline turboquant_int2 --resume --execute
python scripts/run_subitkv.py metrics --model wan2_2_5b --run-id demo --resume --execute
```

Supported baseline names are `kivi_int2`, `kivi_int4`, `turboquant_int2`,
`turboquant_int4`, `quantvideo_int2`, and `quantvideo_int4`. The same interface
supports `self_forcing` and `hyworldplay_8b`; HY needs conditioning images.
`--baseline` is independent of `--budget` and requires no calibration bundle.
Omit `--execute` to inspect a dry-run.

Codec tests cover real round trips, packed payloads, preservation of the random
stream, and append-only history; launch tests cover all model/method pairs.
Restoring these paths does not itself validate full checkpoint-backed videos
or reproduce benchmark results.
