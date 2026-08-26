# HY-WorldPlay 8B MovieGen I2V 20/10/973

本流程是 Wan/Self-Forcing 主协议之外的独立 HY-WorldPlay 8B 实验。所有命令
默认只打印计划；只有把 `--execute` 放在子命令之前才会写文件或启动 GPU。

## 1. 固定协议

- 数据身份：MovieGen `dataset_row`，不是 prompt 字符串。
- 划分：train20、dev10、test973，合计覆盖 1003 行且不重叠。
- test973 保留 5 对重复 prompt；不会因文本去重而少算样本。
- conditioning image：逐 shard 流式解码 `test_with_generations` 视频第 0 帧。
- 生成：157 帧、848×464、24 FPS、4 steps、BF16、seed 123。
- attention/cache：append-only、per-chunk 编码、full-history attention、KV CPU
  offload；文本编码后组件 offload，VAE 前释放 Transformer/KV。
- 方法：BF16；Ours avg1/2/3/4；KIVI、TurboQuant、QuantVideo INT2/INT4。

Ours 的 nominal bit 只指每个 K 流和 V 流的 residual 调度。RVQ index、
early-stop depth、scale 和 metadata 全部进入 dynamic payload；共享 codebook
分别按不含、单视频一次、973 视频摊销三种口径报告。

## 2. 环境

```bash
cp .env.example .env
```

至少设置以下绝对路径：

```bash
QVG_DATA_ROOT=/shared/iclr2026/data
QVG_OUTPUT_ROOT=/shared/iclr2026/results
QVG_MOVIEGEN_ROOT=/datasets/movie_gen_video_bench
QVG_HY8B_MODEL=/checkpoints/HunyuanVideo-1.5
QVG_HY_ACTION_CKPT=/checkpoints/HY-WorldPlay/ar_distilled_action_model/model.safetensors
QVG_PYTHON=/envs/hyworldplay/bin/python
QVG_VBENCH=/envs/vbench/bin/vbench
```

检查工具、checkpoint 和 GPU：

```bash
python scripts/main/hyworldplay_i2v.py --run-id hy-main-r1 doctor
python scripts/main/hyworldplay_i2v.py --run-id hy-main-r1 --execute doctor
```

GPU 会立即领取，但必须同时满足：无 compute process、显存使用不超过
512 MiB、利用率不超过 5%。每张卡有独占文件锁；外部进程进入后，该 attempt
会保留并标为 contaminated，然后写入新的 attempt 目录重试。

## 3. 数据准备

```bash
# 只核查 1003 行和 frozen split，不写输出
python scripts/main/hyworldplay_i2v.py --run-id hy-main-r1 prepare-data

# 流式提取 1003 张 PNG 和 manifest
python scripts/main/hyworldplay_i2v.py --run-id hy-main-r1 --execute prepare-data
```

源 MP4 每次只将一行写入自动删除的临时 seekable 文件供 ffmpeg 解码，不复制
到数据输出目录或 Git。manifest 记录
dataset row、shard、prompt/source video/conditioning image SHA-256。

## 4. 训练

```bash
python scripts/main/hyworldplay_i2v.py \
  --run-id hy-main-r1 --gpus 0,1,2,3,4,5,6,7 train

python scripts/main/hyworldplay_i2v.py \
  --run-id hy-main-r1 --gpus 0,1,2,3,4,5,6,7 \
  --resume --execute train
```

训练依次完成：

1. 20 条 BF16 append-only 轨迹；每层每 prompt 在 10 个 span 各采 125
   个位置。RoPE/ProPE 两流使每个 layer/head 最终得到 50k vectors。
2. K/V 独立的 54 layers × 16 heads、no-norm L16C64 codebook；20 次
   K-means、seed 42。
3. threshold 0.0629、depth 4–16 的 early stop；K frame mean + Hadamard
   G128 + routed Gaussian；V progressive sign-adaptive G128。
4. train20 第二遍 attention calibration，分别测 K-only/V-only output error。
5. 从同一校准产物各自通过 MILP 冻结容量 10/20/30/40 的 avg1/2/3/4
   K/V profile；旧 span 只能降精度，终态必须恰好用满容量。

所有 codebook、capture、profile 都位于 `QVG_DATA_ROOT`，被 `.gitignore`
排除。每个阶段有 create-only `run_info.json` 和 SHA-256。

## 5. dev10 gate

```bash
python scripts/main/hyworldplay_i2v.py \
  --run-id hy-main-r1 --gpus 0,1,2,3,4,5,6,7 \
  --resume --execute gate
```

gate 实际生成 10×11 条完整视频，并检查：157 帧、848×464、24 FPS、
conditioning image 和输入 hash、首 chunk latent hash、append-only payload
audit，以及所有有限指标。四个 Ours 配置都必须满足 PSNR≥25、SSIM≥0.80、
LPIPS≤0.25。dev10 只用于 gate，不参与训练或 test973 汇总。

## 6. test973 生成与跨机器分片

单机：

```bash
python scripts/main/hyworldplay_i2v.py \
  --run-id hy-main-r1 --gpus 0,1,2,3,4,5,6,7 \
  --resume --execute generate --split test --methods all
```

跨 4 台机器稳定分片（每台共享相同输出文件系统和 run-id）：

```bash
python scripts/main/hyworldplay_i2v.py \
  --run-id hy-main-r1 --gpus 0,1,2,3,4,5,6,7 \
  --shard-index 0 --num-shards 4 --resume --execute \
  generate --split test --methods all
```

其余机器分别使用 shard-index 1、2、3。分片规则固定为
`dataset_row % num_shards`。中断后重复同一命令并加入 `--resume`；成功 attempt
跳过，失败/OOM/污染 attempt 永不覆盖或删除。

## 7. PSNR/SSIM/LPIPS 与 VBench

```bash
python scripts/main/hyworldplay_i2v.py \
  --run-id hy-main-r1 --gpus 0,1,2,3,4,5,6,7 \
  --shard-index 0 --num-shards 4 --resume --execute \
  evaluate --split test --methods all
```

Similarity 使用相同 row/image/prompt/seed 的 BF16 reference，严格跳过第 0
帧，只比较 156 帧：PSNR 先逐 RGB 帧算 dB 再平均；SSIM 为 11×11 RGB；
LPIPS 为 VGG，`[0,1]→[-1,1]`。保存逐帧 JSONL、逐 chunk、逐视频和宏平均。

VBench 仅运行 Subject Consistency、Background Consistency、Imaging Quality、
Aesthetic Quality。每条输入先解码并去掉第 0 帧，再以 lossless H.264 写出统一
156 帧副本。每个维度保存官方原始 JSON；聚合前必须证明 973 个 row 各出现
一次。调试 Similarity 时可加 `evaluate ... --skip-vbench`。

## 8. 汇总与一键运行

```bash
python scripts/main/hyworldplay_i2v.py \
  --run-id hy-main-r1 --resume --execute aggregate
```

任一方法少于 973 条成功视频、Similarity row 覆盖不是 973/973，或者 VBench
维度不完整时，聚合器必须失败，不产生静默均值。

完整流程：

```bash
# 安全 dry-run
python scripts/main/hyworldplay_i2v.py --run-id hy-main-r1 all

# prepare → doctor → train → dev gate → 10,703 videos → metrics → aggregate
python scripts/main/hyworldplay_i2v.py \
  --run-id hy-main-r1 --gpus 0,1,2,3,4,5,6,7 \
  --resume --execute all
```

正式输出在
`${QVG_OUTPUT_ROOT}/hyworldplay-moviegen-i2v-20-973/<run-id>/`；训练资产在
`${QVG_DATA_ROOT}/hyworldplay-moviegen-i2v-20-973/artifacts/`。视频、KV dump、
checkpoint、codebook、结果和日志均不进入 Git。
