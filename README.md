# SubBitKV — ICLR2026_RVQ

**Public workflow:** model download → BF16 samples/KV capture → codebook,
importance-weight and schedule calibration → SubBitKV inference → metrics,
including TWE.

The five-stage entrypoint supports Wan2.2 5B, Self-Forcing and HY-WorldPlay 8B,
with nominal residual budgets **0 / 0.2 / 0.5 / 1 / 2 / 3 / 4**. Hyperparameters
live in [`configs/subbitkv.yaml`](configs/subbitkv.yaml).
Scheduling reserves RVQ Base storage at each step, converts the remaining bytes
into residual bit counts, and jointly optimizes all seven budget schedules.

To generate a schedule directly from the saved HY train4 calibration constants,
without model downloads, calibration or GPU dependencies:

```bash
python scripts/subbitkv_schedule.py --budget 0.5 --output outputs/hy_avg0p5_dp.json
```

This entrypoint defaults to per-step DP; `--solver milp` selects joint optimization
(requires SciPy). The saved constants contain measured B1–B8 losses and age weights.
B0 is unmeasured: sub-bit schedules explicitly mark the `D(0)=4*D(1)` estimate.

```bash
pip install -e '.[release,dev]'
# Read-only plan; add --execute to run a stage.
python scripts/subbitkv.py download-models --model wan2_2_5b
python scripts/subbitkv.py collect         --model wan2_2_5b
python scripts/subbitkv.py calibrate       --model wan2_2_5b
python scripts/subbitkv.py infer           --model wan2_2_5b
python scripts/subbitkv.py metrics         --model wan2_2_5b
```

See the **[public setup and reproduction guide](docs/subbitkv.md)** for inputs,
asset provenance, resumable runs, metric definitions and validation scope.
Fresh public calibration is separate from the frozen historical protocols below.

## Historical experiment protocols

组内复现仓库：在 **Wan2.2 5B** 和 **Self-Forcing** 上训练 progressive
RVQ，并运行完整 MovieGenVideoBench 1003-prompt KV-cache 压缩主实验。

> 这是给实验协作者使用的入口。请优先使用 `scripts/main/`，不要从
> `scripts/` 中随机选择历史研究脚本。

## HY-WorldPlay 8B I2V：20 train / 10 dev / 973 test

HY-WorldPlay 的独立 MovieGen I2V 主协议已经冻结在
[`configs/main/hyworldplay-moviegen-i2v-20-973.yaml`](configs/main/hyworldplay-moviegen-i2v-20-973.yaml)。
它按 `dataset_row`（不是 prompt 文本）划分数据，运行 BF16、Ours
avg1/2/3/4，以及 KIVI、TurboQuant、QuantVideo 的 INT2/INT4，共 11 种方法。

配置好 `.env` 后先检查完整 dry-run：

```bash
python scripts/main/hyworldplay_i2v.py \
  --run-id hy-main-r1 \
  --gpus 0,1,2,3,4,5,6,7 \
  all
```

确认计划后，以下命令会依次准备 conditioning images、训练、运行 dev10
gate，并仅在 gate 通过后启动 973×11 个正式视频及评测：

```bash
python scripts/main/hyworldplay_i2v.py \
  --run-id hy-main-r1 \
  --gpus 0,1,2,3,4,5,6,7 \
  --resume \
  --execute \
  all
```

完整的逐阶段命令、跨机器分片、恢复方式、指标定义和输出布局见
[`docs/hyworldplay_i2v_20_973.md`](docs/hyworldplay_i2v_20_973.md)。

## 实验矩阵

两个模型使用相同的 1003 个测试 prompt。每个模型运行以下 11 个配置：

| 方法 | 配置 |
| --- | --- |
| BF16 | generation-time KV CPU offloading |
| RVQ (Ours) | average residual budget 1 / 2 / 3 / 4 bit |
| TurboQuant | K2/V2、K4/V4 |
| KIVI | INT2、INT4 |
| QuantVideo | pre-RoPE INT2、INT4 |

RVQ 1/2/3/4 共用一套 progressive L16C64 K/V codebook，只改变运行时
bit budget，不需要分别训练四套 codebook。

主协议位于 [`configs/main/moviegen-1003.yaml`](configs/main/moviegen-1003.yaml)，
prompt 固定为 [`assets/MovieGenVideoBench.txt`](assets/MovieGenVideoBench.txt)。

## 1. 克隆和安装

```bash
git clone https://github.com/linbowen0719-beep/ICLR2026_RVQ.git
cd ICLR2026_RVQ

conda create -n iclr2026-rvq python=3.12 -y
conda activate iclr2026-rvq

pip install -e ".[all]"
pip install vbench
```

建议使用 Linux、CUDA GPU、PyTorch 2.8。单个生成任务使用一张 GPU；正式
配置在 RTX 6000 Ada 48 GiB 上验证。Flash Attention 和 CUDA 版本说明见
[`docs/installation.md`](docs/installation.md)。VBench 安装方式来自
[VBench 官方仓库](https://github.com/Vchitect/VBench)。

## 2. 配置本机路径

```bash
cp .env.example .env
```

编辑 `.env`，全部使用绝对路径：

```bash
QVG_CKPT_ROOT=/shared/checkpoints/iclr2026-rvq
QVG_DATA_ROOT=/shared/data/iclr2026-rvq
QVG_CODEBOOK_ROOT=/shared/codebooks/iclr2026-rvq
QVG_OUTPUT_ROOT=/shared/results/iclr2026-rvq
QVG_CACHE_ROOT=/shared/cache/iclr2026-rvq
HF_HOME=/shared/cache/iclr2026-rvq/huggingface
```

`.env`、checkpoint、codebook、视频和结果都不会进入 Git。

## 3. 下载模型

```bash
bash scripts/HY-WorldPlay/download_models.sh
bash scripts/Self-Forcing/download_models.sh
```

下载后应满足：

```text
${QVG_CKPT_ROOT}/
├── HY-WorldPlay/
│   ├── wan_transformer/config.json
│   ├── wan_transformer/diffusion_pytorch_model.safetensors
│   └── wan_distilled_model/model.pt
└── Self-Forcing/
    ├── Wan2.1-T2V-1.3B/
    └── self_forcing_dmd.pt
```

## 4. 训练 RVQ

RVQ 训练是正式运行流程的一部分。它使用冻结的 20 条 calibration prompt；
RVQ 1/2/3/4 共用训练得到的 progressive L16C64 codebook。先准备逐层 KV dump：

```text
<work-root>/kv_dump/train/<prompt-index>/layer_<layer>.pt
```

然后先 dry-run，再执行 K/V 训练：

```bash
python scripts/main/train_rvq.py \
  --model wan2_2_5b \
  --work-root /shared/calibration/wan2_2_5b \
  --gpus 0,1,2,3,4,5,6,7

python scripts/main/train_rvq.py \
  --model wan2_2_5b \
  --work-root /shared/calibration/wan2_2_5b \
  --gpus 0,1,2,3,4,5,6,7 \
  --execute
```

Self-Forcing 将 `--model` 改为 `self_forcing` 并使用独立 work-root。训练入口
依次训练 K/V，记录 hash 与 `run_info.json`，且不会覆盖已有 codebook。当前入口
从已采集 KV dump 开始；KV dump 不进入 Git。

## 5. 组装运行资产

训练后的 codebook、routing、tree 和 temporal profile 约 1.1 GB，不保存在
GitHub。训练完成后将 codebook 与固定 routing/tree/temporal profile 组装为
`paper_main_v1`；只负责生成的协作者也可直接获取该 bundle，并放到：

```text
${QVG_DATA_ROOT}/paper_main_v1/
├── manifest.json
├── shared/
│   ├── tree_params.pt
│   ├── progressive_v_params.pt
│   └── temporal_profile.json
├── wan2_2_5b/
│   ├── routing.json
│   ├── nonorm_k_l16c64_codebooks/
│   └── nonorm_v_l16c64_codebooks/
└── self_forcing/
    ├── routing.json
    ├── nonorm_k_l16c64_codebooks/
    └── nonorm_v_l16c64_codebooks/
```

不要修改 bundle 内的文件名。`doctor` 会验证 prompt、文件数量和 SHA-256。

## 6. 运行环境检查

```bash
python scripts/run_paper_main.py doctor --full-hash
```

只有看到：

```json
{"status": "pass"}
```

才开始正式生成。如果检查失败，根据 `errors` 中给出的具体路径补齐文件。

## 7. 先检查运行计划

下面的命令只打印任务，不创建输出，也不加载模型：

```bash
python scripts/main/run_moviegen.py \
  --run-id main-YYYYMMDD-r1 \
  --gpus 0,1,2,3,4,5,6,7
```

正式矩阵共有 2 个模型 × 11 个方法 × 1003 个 prompt，即 22,066 个视频任务。
每个任务使用一张 GPU，调度器会自动填满 `--gpus` 指定的设备。

## 8. 正式生成 MovieGenVideoBench-1003

```bash
python scripts/main/run_moviegen.py \
  --run-id main-YYYYMMDD-r1 \
  --gpus 0,1,2,3,4,5,6,7 \
  --execute
```

只运行一个模型：

```bash
python scripts/main/run_moviegen.py \
  --run-id main-YYYYMMDD-r1 \
  --model wan2_2_5b \
  --gpus 0,1,2,3,4,5,6,7 \
  --execute
```

`--model` 也可以设置为 `self_forcing`。相同 `run-id` 可以安全续跑：已有
`completed.json` 的任务会跳过。改变 prompt、seed、方法或配置时必须换一个
新的 `run-id`。

## 9. Benchmark 和论文表格

两套模型的 BF16 和压缩视频全部完成后运行：

```bash
# 先 dry-run
python scripts/main/run_benchmark.py \
  --run-id main-YYYYMMDD-r1 \
  --gpus 0,1,2,3,4,5,6,7

# 确认后正式执行
python scripts/main/run_benchmark.py \
  --run-id main-YYYYMMDD-r1 \
  --gpus 0,1,2,3,4,5,6,7 \
  --execute
```

Benchmark 按顺序执行：

1. 同 prompt、同 seed BF16 reference 的 PSNR / SSIM / LPIPS；
2. VBench-10；
3. 完整物理 KV 压缩率；
4. E2E、encode、decode 时间；
5. peak GPU memory 和 KV payload MiB；
6. 完整性校验与 CSV/LaTeX 汇总。

聚合器缺少任何一个 method、prompt、quality 文件或 VBench dimension 时都会
失败，不会对不完整结果静默取平均。

## 10. 输出位置

```text
${QVG_OUTPUT_ROOT}/main-YYYYMMDD-r1/
├── pipeline_run_info.json
├── wan2_2_5b/
│   ├── videos/<method>/...
│   ├── similarity_postcompression/...
│   ├── vbench_flat_v2/...
│   └── summary.csv
├── self_forcing/
│   ├── <method>/prompt_####_seed_0/
│   │   ├── 0-0_ema.mp4
│   │   ├── metrics.json
│   │   ├── quality_cross_model.json
│   │   ├── resolved_config.json
│   │   └── run_info.json
│   └── vbench10/...
└── aggregate/
    ├── main_results.csv
    ├── per_prompt.csv
    ├── main_results.tex
    └── run_info.json
```

最终论文表格是 `aggregate/main_results.tex`，机器可读主结果是
`aggregate/main_results.csv`。

## 主要代码结构

```text
scripts/main/
├── train_rvq.py                  # RVQ K/V codebook 训练
├── run_moviegen.py               # 11-method MovieGen 生成矩阵
├── run_benchmark.py              # quality、VBench、runtime、memory
├── export_main_table_bundle.py   # 导出非 Git 资产
└── aggregate_paper_main.py       # 完整性检查和论文表格

quant_videogen/                   # RVQ、KIVI、TurboQuant、QuantVideo codec
experiments/HY-WorldPlay/         # Wan2.2 5B 集成
experiments/Self-Forcing/         # Self-Forcing 集成
configs/main/                     # 冻结实验协议
assets/                           # 固定 20 条训练 prompts 与 1003 条生成 prompts
tests/                            # CPU 和可选 CUDA tests
```

更详细的逐阶段说明见
[`docs/main_table_reproduction.md`](docs/main_table_reproduction.md)。

## 测试

```bash
pytest -q \
  tests/test_paper_main.py \
  tests/test_reproduce.py \
  tests/test_repository_hygiene.py
```

修改 quantize / encode / decode / restore 路径后，还必须执行对应 CUDA smoke 和
round-trip test。

## 数据与安全约定

- 不提交 checkpoint、KV dump、codebook、视频、完整结果或日志。
- 不覆盖旧的 `run-id`。
- 内存统一报告 MiB；压缩率必须包含完整压缩表示和共享 codebook。
- reconstruction metric 必须保持 K/V accounting 和 prompt/seed 一致。
- 每次运行必须保留 `run_info.json`。

第三方模型代码和许可证说明见 [`THIRD_PARTY.md`](THIRD_PARTY.md)。
