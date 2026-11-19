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

## 结果

| 阶段 | 训练数据 | 训练 |
|---|---|---|
| A1 | 135 条、35 个任务，每轮生成 token 中位数 460 | 48 步，loss 0.26，1 h 13 min |
| A2 | 48 条、19 个任务，中位数 563 | 18 步，loss 0.22 → 0.15 |
| A3 | 74 条、22 个任务，中位数 525 | 已完成 |

LIBERO 40 局（trial 0），每格为成功局数 / 每轮生成 token；括号内为与 PEAR-27B 按 episode 配对时“仅该模型成功 / 仅 PEAR-27B 成功”的局数。PEAR-27B 的 xhigh 为 FP8，其余均为 BF16。

| 评测档位 | PEAR-27B | A1 | A2 | A3 |
|---|---|---|---|---|
| xhigh | 29 / 3643 | 28 / 1193（1 / 2） | 28 / 1249（2 / 3） | 32 / 1131（3 / 0） |
| medium | 11 / 947 | 7 / 749（4 / 8） | 8 / 627（4 / 7） | 12 / 597（7 / 6） |
| low | 12 / 1029 | 4 / 685（0 / 8） | 13 / 615（5 / 4） | 10 / 647（8 / 10） |

A3 在三个档位上与 PEAR-27B 的成功局数差距都在 3 局以内；xhigh 档每轮生成 token 减少 69%，降幅几乎全部来自 A1。A1 在 low 档明显下降（4/40），A2、A3 恢复到 PEAR-27B 的水平，原因未查明。

### 泛化（xhigh，未参与训练的扰动场景）

评测子集按扰动类型分层，从 PEAR-27B 已有的 rollout 中抽取，与其按 episode 配对。

| 基准 | 局数 | 成功（A3 / PEAR-27B） | 配对差异 | 每轮生成 token（A3 / PEAR-27B） |
|---|---:|---|---|---|
| LIBERO-Plus | 56 | 32 / 35 | 5 / 8，p = 0.58 | 1656 / 2155（−23%） |
| LIBERO-Pro | 40 | 21 / 21 | 2 / 2，p = 1.00 | 1904 / 1987（−4%） |

A1 的结果相近：Plus 33/35、Pro 22/21，每轮 token 分别减少 17% 与 15%。成功率与 PEAR-27B 持平，token 降幅明显小于 LIBERO 上的 69%，结论待定。

## 未采用的方法

| 方法 | 原因 |
|---|---|
| 截断思维链 + 蒙特卡洛续写（JET、Double-Rollout、ConCISE） | 需新写打分脚本并跑数小时推理，时间成本高于本方法 |
| 带长度惩罚的强化学习（ALP、DLER、LACONIC、L1） | 每步需在线仿真，27B 上成本过高 |
