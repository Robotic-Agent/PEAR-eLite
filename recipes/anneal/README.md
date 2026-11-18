# 思维链退火：PEAR-27B xhigh → medium → low

PEAR-27B 的模板只支持 xhigh、medium、low 三档。每个阶段在当前模型的自采样中，每个任务取生成 token 最少的 K 条成功轨迹，在该档位上做 LoRA SFT，合并后作为下一阶段的起点（最短正确样本自训练，Ho 等 2025；SmartThinker 第一阶段）。

| 阶段 | 起点 | 采样 | 数据 | 产出 |
|---|---|---|---|---|
| A1 | PEAR-27B | 已有 xhigh rollout | 每个任务最短 4 条 | `Qwen3.8-27B-PEAR-A1-xhigh` |
| A2 | A1 | A1 在 medium，LIBERO trial 1–8（320 局） | 同上 | `Qwen3.8-27B-PEAR-A2-medium` |
| A3 | A2 | A2 在 low，trial 1–8 | 同上 | `Qwen3.8-27B-PEAR-A3-low` |

评测：每个模型在各档位跑 40 局（trial 0），与未训练的 PEAR-27B 比较成功率与每轮生成 token。

## 入口

```bash
# 数据：每个任务最短 K 条成功轨迹，去掉 trial 0
python scripts/prepare_effort_sft.py --source "1:<采样 run>[,...]" --per-task 4 --exclude-trials 0 \
  --max-response-tokens 36000 --out train.parquet
# LoRA SFT（r 64，lr 5e-5，3 个 epoch，global batch 8）
WORK_DIR=<scripts/install.sh 的工作目录> REASONING_EFFORT=medium MODEL=<上一阶段模型> RUN=<输出> TRAIN_DATA=train.parquet EPOCHS=3 GLOBAL_BATCH=8 LR=5e-5 \
  bash examples/models/qwen3_8_27b_pear/train/run_effort_sft_lora_5090.sh
# 合并
python tools/lora/merge_lora_shards.py <基座> <adapter 目录> <输出目录>
```

cluster 上的串联：<internal job scripts>（单阶段为 <internal job scripts>）。

## 结果

| 阶段 | 训练 | 评测 |
|---|---|---|
| A1 | 135 条、35 个任务，每轮生成 token 中位数 460（全部 xhigh 成功轨迹 654）；48 步，loss 0.26，1 h 13 min | 结论待定 |
| A2 | 运行中 | — |
| A3 | 未开始 | — |

## 未采用的方法

| 方法 | 原因 |
|---|---|
| 截断思维链 + 蒙特卡洛续写（JET、Double-Rollout、ConCISE） | 需新写打分脚本并跑数小时推理，时间成本高于本方法 |
| 带长度惩罚的强化学习（ALP、DLER、LACONIC、L1） | 每步需在线仿真，27B 上成本过高 |
