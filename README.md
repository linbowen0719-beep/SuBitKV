<div align="center">

# QuantVideoGen

**Quantized KV-cache compression for autoregressive video generation**

[Project page](https://svg-project.github.io/qvg/) ·
[Paper](https://arxiv.org/abs/2602.02958) ·
[Reproduction guide](docs/reproduction.md) ·
[Main-table run](docs/main_table_reproduction.md)

</div>

QuantVideoGen provides Triton-backed KV-cache quantization and research
integrations for LongCat-Video, Self-Forcing, and HY-WorldPlay. The repository
separates supported reproduction entrypoints from historical research scripts
so a fresh checkout can be inspected and configured without access to the
original authors' filesystem.

## Supported integrations

| Model | BF16 integration | INT2 QVG | Smoke config | Main config |
| --- | :---: | :---: | --- | --- |
| LongCat-Video | Yes | Yes | `longcat` | `longcat-qvg-int2` |
| Self-Forcing | Yes | Yes | `selfforcing` | `selfforcing-qvg-int2` |
| HY-WorldPlay | Yes | Yes | `hyworldplay` | `hyworldplay-qvg-int2` |

The six configurations above are the supported interface. Other files under
`scripts/` and `experiments/` are retained for research traceability and may
require additional datasets or codebooks.

## Quick start

```bash
git clone https://github.com/svg-project/Quant-VideoGen.git
cd Quant-VideoGen

conda create -n qvg python=3.12 -y
conda activate qvg
pip install -e ".[all]"

cp .env.example .env
cp configs/paths.example.yaml configs/paths.local.yaml
# Edit .env or configs/paths.local.yaml for this machine.

python scripts/reproduce.py list
python scripts/reproduce.py doctor --config configs/paths.local.yaml
python scripts/reproduce.py run --suite smoke --model longcat --dry-run
python scripts/reproduce.py run --suite smoke --model longcat
```

`doctor` reports every missing command, checkpoint, or input together with its
preparation command. A smoke run creates a new timestamped directory under
`QVG_OUTPUT_ROOT`; it never reuses an existing run directory.

Inspect a full experiment before explicitly authorizing it:

```bash
python scripts/reproduce.py run \
  --suite main \
  --experiment longcat-qvg-int2 \
  --dry-run

python scripts/reproduce.py run \
  --suite main \
  --experiment longcat-qvg-int2 \
  --execute
```

Main runs require `--execute`. Repository setup does not automatically launch
long evaluations.

For the full Wan2.2 5B + Self-Forcing paper table (22 rows, 10 prompts each),
use the lifecycle-oriented multi-GPU entrypoints:

```bash
python scripts/run_paper_main.py doctor --full-hash
python scripts/main/run_moviegen.py --run-id main-YYYYMMDD-r1 --gpus 0,1,2,3,4,5,6,7
python scripts/main/run_benchmark.py --run-id main-YYYYMMDD-r1 --gpus 0,1,2,3,4,5,6,7
```

Asset transfer and launch instructions are in the
[main-table guide](docs/main_table_reproduction.md).

## Repository layout

```text
quant_videogen/       Importable compression and kernel code
experiments/          Vendored, modified model integrations and protocols
configs/smoke/        Supported minimal GPU runs
configs/main/         Supported full integration runs
scripts/reproduce.py  list / doctor / dry-run / execute interface
scripts/main/         RVQ training / MovieGen generation / benchmark modules
scripts/eval/         Reusable evaluation entrypoints
docs/                 Setup, data, testing, reports, and research index
tests/                CPU and opt-in CUDA tests
```

Checkpoints, data, codebooks, generated videos, logs, and complete results are
local-only and excluded from Git. See [repository layout](docs/repo_layout.md)
and [data preparation](docs/data_and_checkpoints.md).

## Reproducibility

Every supported run stores the resolved configuration, per-step logs, and a
`run_info.json` containing the Git commit, stable config hashes, host/GPU,
seed, prompt list, timestamp, and optional codebook hash. Expected output files
are checked before a run is marked complete.

Installation variants and CUDA notes are in [installation](docs/installation.md).
Exact commands and resource expectations are in
[reproduction](docs/reproduction.md). Testing tiers are documented in
[testing](docs/testing.md).

## Citation

```bibtex
@article{xi2026quant,
  title={Quant VideoGen: Auto-Regressive Long Video Generation via 2-Bit KV-Cache Quantization},
  author={Xi, Haocheng and Yang, Shuo and Zhao, Yilong and Li, Muyang and others},
  journal={arXiv preprint arXiv:2602.02958},
  year={2026}
}
```

## License

Original QuantVideoGen code is Apache-2.0. Vendored integrations retain their
upstream licenses. In particular, HY-WorldPlay uses the Tencent HY-WorldPlay
Community License with additional restrictions. Read [THIRD_PARTY.md](THIRD_PARTY.md)
before redistributing the repository.
