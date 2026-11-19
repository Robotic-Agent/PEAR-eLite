# PEAR-eLite：PEAR-27B 思维链退火、小模型蒸馏与 slime 适配

本仓库基于 [mit-han-lab/pear](https://github.com/mit-han-lab/pear) 提交 `f32aa1f`（前身为 pear-lite），目标是把 Qwen3.8-27B-PEAR-LIBERO_only（下称 PEAR-27B）轻量化到多数实验室能研究的规模：缩短它的思维链，并把它蒸馏到 Qwen3.5-4B/9B 学生模型。verl 配方在 8 × RTX 5090（32 GB，SM120）上运行，slime 配方在 8 × H100 上运行。LIBERO 结果为初步验证，后续计划换用更不易过拟合的环境（包括游戏强化学习环境，见第 9 节）。

第 1–7 节为 pear-lite 阶段的记录；第 8 节起为 PEAR-eLite 新增内容。

## 1. 相对上游的改动

| 提交 | 内容 |
|---|---|
| verl 补丁 0044 | 单个 VLM 样本的 3D position ids 在 tensordict 中丢失 ragged 维度，按序列维重建 |
| verl 补丁 0045 | `VERL_SKIP_QAT_PATCHES=1`：量化教师（不接收权重更新）跳过 QAT 权重加载补丁 |
| verl 补丁 0046 | `attn_implementation=sdpa` 时补上 Ulysses all-to-all（见第 2 节） |
| `pear/trainer/engine.py` | 分块 AdamW（`StreamedAdamW`）不再依赖 GPU 后端；块大小由 `PEAR_STREAMED_ADAMW_CHUNK_BYTES` 设置，32 GB 卡用 256 MiB |
| `scripts/start_ray_node.sh` | `trainer_rollout` 角色：单节点同时承载训练、教师与 rollout 资源池 |
| `scripts/install/verl/install.sh` | `TRANSFER_QUEUE_SRC`：无 GitHub 访问的机器使用本地 TransferQueue 源码安装 |
| `examples/models/qwen3_5_4b_pear/` | 学生 SFT（`train/run_sft_full_h100.sh`，经参数覆盖后在 5090 上运行）与 OPD（`distill/run_opd_5090.sh`） |
| `examples/models/qwen3_8_27b_pear/train/run_effort_sft_lora_5090.sh` | PEAR-27B 按推理强度的 LoRA SFT（RFT） |
| `scripts/*.py` | 轨迹 → 多轮多模态 SFT 数据（`prepare_effort_sft.py`、`pear_episode_sft_dataset.py` 等） |
| `tools/` | 不依赖 Ray 的 LIBERO 评测、配对比较、NVFP4 教师量化与核对、LoRA 合并 |
| `tests/gpu/test_ulysses_sdpa.py` | Ulysses SDPA 与单卡 SDPA 的输出和梯度对比 |

verl 补丁通过 PEAR 自身机制应用：`bash 3rdparty/verl/scripts/checkout_verl.sh` 构建 `8e03c039 + 46 patches`，`--check` 校验通过，不需要 `VERL_CHANGE_IN_PROGRESS=1`。

## 2. 已发现的问题：SDPA 下的 Ulysses 序列并行

5090 上未安装 flash-attn，配方使用 `attn_implementation=sdpa`。verl 只在 `_flash_attention_forward` 外插入 Ulysses all-to-all，因此 `ulysses_sequence_parallel_size > 1` 且使用 SDPA 时，每个 rank 的全注意力层只在本 rank 的 1/sp 序列切片内计算注意力。补丁 0046 在 `ALL_ATTENTION_FUNCTIONS["sdpa"]` 外实现 all-to-all，按 packed 序列边界分段做因果 SDPA。

| 检查 | 结果 |
|---|---|
| `tests/gpu/test_ulysses_sdpa.py`（输出与 dq/dk/dv） | 与单卡参考一致，`ULYSSES_SDPA_OK` |
| OPD 8k，SP=4，修复前（run 008） | `rollout_corr/kl` 0.254，训练/rollout log-ppl 1.05 / 0.79 |
| OPD 8k，SP=4，修复后（run 010） | `rollout_corr/kl` 0.0003，log-ppl 0.7948 / 0.7945 |
| OPD 8k，SP=1（run 004，参照） | `rollout_corr/kl` 0.0004 |

受影响的已完成训练：

- 4B 学生全参数 SFT（SP=8）
- PEAR-27B RFT LoRA（R1，`run_effort_sft_lora_5090.sh` 中 `ulysses_sequence_parallel_size=8`）

两者需在修复后重训。第 3 节中这两项结果均来自修复前的训练。

## 3. 结果（实测）

### 3.1 RFT 思维链缩短（PEAR-27B，R1）

数据：PEAR-27B 在 LIBERO 训练任务上以 medium 强度采样，`prepare_effort_sft.py` 对每个任务保留 token 数最少的一条成功轨迹；LoRA rank 64，在 8 × 5090 上训练。评测与未训练的 PEAR-27B 按相同 episode（suite、task、trial）配对比较。

| 推理强度 | 配对 episode | 成功率（R1 / 基线） | 每轮生成 token（R1 / 基线） | 状态 |
|---|---:|---|---|---|
| medium | 79 | 30.4% / 25.3%，p = 0.541 | 803 / 1020（−21%） | token 减少已确认；成功率差异不显著 |
| xhigh | 32 | 87.5% / 87.5% | 2008 / 2088（−4%） | 无明显缩短 |
| low | — | — | — | 评测排队中 |

注意：R1 训练时存在第 2 节的注意力问题，修复后重训的结果结论待定。

### 3.2 学生 SFT（Qwen3.5-4B/9B）

- 学生模型目录 `Qwen3.5-{4B,9B}-pear` = Qwen3.5 原始权重 + PEAR-27B 的 `chat_template.jinja`、`tokenizer.json`、`tokenizer_config.json`（与教师 token id 一致）。
- 未训练的 4B：40 个 episode，成功 0，其中 38 个因输出无法解析而结束。
- 4B S-xhigh 全参数 SFT（388 条成功轨迹，最长 64k token，3 个 epoch，72 步，loss 约 1.25 → 0.70）：训练时存在第 2 节的问题。
- 9B S-xhigh 全参数 SFT（同一数据，72 步，loss 约 0.87 → 0.5）：未受第 2 节问题影响。

LIBERO 评测（xhigh，BF16，温度 0，40 个 episode）：

| 模型 | 成功 | 结束原因（env / budget / parse） | 每轮生成 token |
|---|---:|---|---:|
| Qwen3.5-4B-pear（未训练） | 0/40 | parse 38，其余 2 | 1444 |
| Qwen3.5-9B-pear（未训练） | 0/40 | 11 / 1 / 28 | — |
| 4B S-xhigh | 15/40（37.5%） | 35 / 4 / 1 | 4602 |
| 9B S-xhigh | 22/40（55.0%） | 36 / 4 / 0 | 2499 |

9B 与 4B 学生按相同 episode 配对（33 个）：22/33 对 15/33，仅 9B 成功 10 个，仅 4B 成功 3 个，McNemar p = 0.092，差异结论待定。

### 3.3 OPD（PEAR-27B → Qwen3.5-4B）

| 配置 | 结果 |
|---|---|
| 8k，BF16 教师 TP4，训练 2 卡（run 004） | 端到端完成；distill loss 0.178，`rollout_corr/kl` 0.0004，107 s/步 |
| 8k，训练 SP=4，修复后（run 010） | distill loss 0.176，grad norm 11.6，`rollout_corr/kl` 0.0003，98.7 s/步，显存峰值 13.1 GB 已分配 / 22.5 GB 已保留 |
| 32k，NVFP4 教师 TP2，训练 4 / rollout 2 / 教师 2（run 007） | 第 1 步完成，显存 15.8 / 27.6 GB；第 2 步 DataLoader 进程被 SIGKILL，原因未查明；该运行在修复前，需复核 |

NVFP4A16 教师（`tools/teacher/quant_nvfp4a16.py`，lm_head、视觉塔、Gated DeltaNet 投影保留 BF16）与 BF16 教师在同一批 rollout 文本上的对比：top-1 一致率 0.9455，top-32 重合率 0.9108，|Δlogp| 中位数 0.023、均值 0.142、p99 2.25，KL 0.040。

## 4. 运行

环境按上游 README 用 `scripts/install.sh` 构建（`WORK_DIR` 为其工作目录），然后：

```bash
# 学生 SFT（5090：SP=8，优化器状态放在主机）
WORK_DIR=... RUN=... TRAIN_DATA=<prepare_effort_sft.py 输出> MODEL=$WORK_DIR/models/Qwen3.5-9B-pear \
MAX_LENGTH=65536 SP=8 TOKENS_PER_GPU=8192 \
  bash examples/models/qwen3_5_4b_pear/train/run_sft_full_h100.sh \
  engine.optimizer_offload=true engine.optimizer_offload_step=true
# 32 GB 卡需 export PEAR_STREAMED_ADAMW_CHUNK_BYTES=268435456

# RFT 数据：每个任务最短的成功轨迹
python scripts/prepare_effort_sft.py --source 1:<eval run dir> --out rft.parquet --max-response-tokens 36000

# OPD，NVFP4 教师
LIBERO_SERVERS_PER_NODE=2 bash scripts/start_ray_node.sh trainer_rollout &
VERL_SKIP_QAT_PATCHES=1 PEAR_STREAMED_ADAMW_CHUNK_BYTES=268435456 \
TRAINER_GPUS=4 ROLLOUT_GPUS=2 TEACHER_GPUS=2 TEACHER_TP=2 \
TEACHER_MODEL=$WORK_DIR/models/Qwen3.8-27B-PEAR-LIBERO_only-NVFP4A16 \
RAY_ADDRESS=auto WORK_DIR=... RUN=... \
  bash examples/models/qwen3_5_4b_pear/distill/run_opd_5090.sh \
  actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=4

# LIBERO 评测与配对比较
WORK_DIR=... bash tools/eval/run_eval.sh <模型目录或 $WORK_DIR/models 下的名称> libero_all.json <输出名>
python tools/analysis/compare_paired.py <标签> <新 run 目录> <基线 run 目录>
```

## 5. 数据与权重

评测轨迹（含录像）、RFT 采样轨迹、SFT 数据、OPD 日志与 rollout 已公开在 ModelScope，可匿名下载：

```bash
modelscope download --dataset candlest/pear-libero-trajectories --local_dir ./pear-libero-trajectories
```

<https://www.modelscope.cn/datasets/candlest/pear-libero-trajectories>（1524 个文件，17.87 GB；每个 run 的帧与录像打包为 `episodes.tar`）。

已公开的权重：[candlest/Qwen3.5-4B-PEAR-LIBERO-SFT-v0](https://www.modelscope.cn/models/candlest/Qwen3.5-4B-PEAR-LIBERO-SFT-v0)、[candlest/Qwen3.5-4B-PEAR-LIBERO-SFT-v2](https://www.modelscope.cn/models/candlest/Qwen3.5-4B-PEAR-LIBERO-SFT-v2)。其余权重（NVFP4 教师、9B 学生、退火模型）尚未公开。

## 6. slime 环境

`env/slime/` 为迁移到 slime / SGLang 的标准化环境：版本全部固定，自带 Python 与 CUDA 13 工具链，CUDA 扩展覆盖 H100、B200、RTX 5090，可打包为目录 tar 或构建为 Docker 镜像。说明见 [`env/slime/README.md`](../env/slime/README.md)。已在 RTX 5090 上通过 `check.py`。

## 7. 后续计划

1. 迁移到 slime / SGLang：依次为 9B SFT、思维链缩短（RFT）、OPD。4B 不再重训。
2. 修复后重训 R1，并重新评测。
3. OPD 在 32k 上复核（修复后）。

## 8. PEAR-eLite 新增内容

### 8.1 学生 SFT v2（Qwen3.5-4B，verl）

| 项目 | 内容 |
|---|---|
| 数据 | PEAR-27B 在 LIBERO 上以 xhigh 采样的 1250 条成功轨迹（2000 个回合中的成功者，去掉 trial 0，最长 64k token，去掉 1 条超长轨迹） |
| 配方 | `examples/models/qwen3_5_4b_pear/train/run_sft_full_h100.sh`（Ulysses SDPA 已修复），3 个 epoch，234 步，lr 1e-5，global batch 16，8 × RTX 5090 |
| 训练 | loss 0.96 → 0.58，每步约 24.5 s，显存约 16 GB/卡 |
| 评测 | 40 局（trial 0）21/40；每轮生成 token 4907；libero_10 为 0/10 |

同一数据训练的 9B v2（verl）为 26/40，与 388 条数据的 9B verl（22/40）配对：仅 v2 成功 7 局，仅 v1 成功 3 局，McNemar p = 0.34。

### 8.2 slime 适配

| 项目 | 状态 |
|---|---|
| 9B SFT（`examples/slime/qwen3_5_9b_pear/run_sft.sh`，8 × H100） | 已完成；40 局 26/40（verl 9B 同数据 22/40） |
| ReOPD（`examples/slime/qwen3_5_9b_pear/run_opd.sh`） | 已跑通训练步：reverse KL 0.148 → 0.127（前 2 步），训练与采样 log-prob 差 0.013，每步约 6.5 分钟（32 轮）；32 轮中 1 轮 token 不一致，原因未查明，已从 loss 中去掉 |
| RTX 5090 | 不可用：Transformer Engine 对 head dim 256 没有可用的注意力后端（cuDNN fused 拒绝该输入，FlashAttention 4 报 `CUDADialectError`） |

ReOPD 的 prompt 是 27B 的一条已记录轨迹截到第 t 个用户轮为止，学生生成第 t 轮，教师对学生的序列打分；思考档位按阶段 xhigh → medium → low，阶段之间从上一阶段的 checkpoint 续训。已知问题：medium / low 阶段的历史轮仍是 xhigh 的思维链，与测试时不一致，结论待定。

### 8.3 三阶段思维链退火（PEAR-27B，LoRA）

模板只支持 xhigh / medium / low 三档。每阶段用上一阶段的模型在该档位采样（LIBERO trial 1–8），每个任务取生成 token 最少的 4 条成功轨迹（`scripts/prepare_effort_sft.py --per-task 4 --exclude-trials 0`），LoRA r 64、lr 5e-5、3 个 epoch，合并后作为下一阶段的起点。单阶段由 `examples/models/qwen3_8_27b_pear/train/run_effort_sft_lora_5090.sh` 训练、`tools/lora/merge_lora_shards.py` 合并。

| 阶段 | 数据 | 状态 |
|---|---|---|
| A1 xhigh | 1251 条 xhigh 成功轨迹中每个任务最短 4 条，135 条、35 个任务；每轮生成 token 中位数 460（全部成功轨迹 654） | 已完成：48 步，loss 0.26，1 h 13 min |
| A2 medium | A1 在 medium 采样 320 局，48 条、19 个任务，中位数 563 | 已完成：18 步，loss 0.22 → 0.15 |
| A3 low | A2 在 low 采样 320 局，74 条、22 个任务，中位数 525 | 已完成 |

xhigh 档 LIBERO 40 局：PEAR-27B 29/40、每轮 3643 token；A3 32/40、1131 token（配对 3 / 0）。各档位逐阶段结果与 LIBERO-Plus / Pro 泛化见 [`recipes/anneal/README.md`](../recipes/anneal/README.md)。

### 8.4 轨迹数据集

`tools/data/build_rollout_release.py` 把 27B 的 rollout 打包为每回合一行的 parquet（完整思维链与动作、状态、两路相机 PNG、GIF 录像）：

| 子集 | 回合 | 成功 | 轮数 |
|---|---:|---:|---:|
| libero | 1873 | 1337 | 25430 |
| libero_plus（未跑完） | 3440 | 2074 | 48584 |
| libero_pro（未跑完） | 1599 | 1043 | 22292 |

LIBERO-Plus / Pro 只作泛化测试，不进学生训练。已上传到 ModelScope 数据集 `candlest/pear-libero-trajectories` 的 `rollouts/` 目录。

## 9. 游戏强化学习环境

`pear/envs/games/` 为游戏环境预留，接口与 LIBERO 相同（`pear.sim.env.StatelessEnv` 或 `StatefulEnv`），说明见 [`pear/envs/games/README.md`](../pear/envs/games/README.md)。目前没有已实现的环境。

