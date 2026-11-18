# SFT：PEAR-27B → Qwen3.5-4B/9B

学生模型目录 `Qwen3.5-{4B,9B}-pear` 为 Qwen3.5 原始权重加 PEAR-27B 的 `chat_template.jinja`、`tokenizer.json`、`tokenizer_config.json`，与教师 token id 一致。损失只计在助手轮（思维链与动作）。

## 数据

```bash
# 成功轨迹 → 多轮多模态 SFT 数据（去掉评测用的 trial 0）
python scripts/prepare_effort_sft.py --source "1:<rollout run>[,<rollout run>...]" --all-successes \
  --exclude-trials 0 --out train.parquet
# 按 token 数过滤（不依赖 verl）
python tools/data/count_filter_tokens.py train.parquet <学生模型目录> 64000 train_le64k.parquet
```

## 训练

| 后端 | 入口 | 说明 |
|---|---|---|
| verl | `examples/models/qwen3_5_4b_pear/train/run_sft_full_h100.sh` | 5090 上 SP=8、优化器状态放在主机；需 Ulysses SDPA 修复（verl 补丁 0046） |
| slime | `examples/slime/qwen3_5_9b_pear/run_sft.sh` | 见 [slime](../slime/README.md) |

## 结果（LIBERO 40 局，trial 0，xhigh）

| 模型 | 数据 | 成功 | 状态 |
|---|---|---:|---|
| 4B v0（Ulysses SDPA 问题修复前） | 388 条 | 15/40 | 已确认 |
| 9B，verl | 388 条 | 22/40 | 已确认 |
| 9B，slime | 388 条 | 26/40 | 已确认 |
| 4B v2，verl | 1250 条 | — | 结论待定 |
