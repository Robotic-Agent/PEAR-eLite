<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/pear_elite_logo_dark.svg">
    <img alt="PEAR-eLite" src="docs/assets/pear_elite_logo_light.svg" width="320">
  </picture>
</p>

<p align="center">
PEAR-27B 的思维链退火、小模型蒸馏与 slime 适配
</p>

---

PEAR-eLite 把 [PEAR](https://github.com/mit-han-lab/pear) 的 27B 机器人推理模型 Qwen3.8-27B-PEAR-LIBERO_only（下称 PEAR-27B）轻量化到多数实验室能训练和评测的规模。仓库提供四类配方：

| 配方 | 内容 | 后端 | 硬件 | 状态 |
|---|---|---|---|---|
| [SFT](recipes/sft/README.md) | PEAR-27B 成功轨迹 → Qwen3.5-4B/9B 学生全参数 SFT | verl、slime | 8 × RTX 5090 / 8 × H100 | 已完成 |
| [OPD](recipes/opd/README.md) | 回放式在线蒸馏（ReOPD）：学生在 27B 轨迹的前缀上生成，教师逐 token 打分，思考档位 xhigh → medium → low | slime | 4–8 × H100 | 4B 运行中 |
| [思维链退火](recipes/anneal/README.md) | PEAR-27B 三阶段自训练：每档位取每个任务最短的成功轨迹做 LoRA SFT，xhigh → medium → low | verl | 8 × RTX 5090 | 已完成 |
| [slime 适配](recipes/slime/README.md) | 版本固定的 slime / SGLang / Megatron 环境，PEAR 多轮多模态轨迹的训练与蒸馏接口 | slime | H100 | 已完成 |

另有两套[游戏环境](pear/envs/games/README.md)：《杀戮尖塔 2》用无渲染模拟器做整局离散决策，Minecraft 用真实第一人称 RGB 做移动目标任务。它们复用 PEAR 的回合与 JSON 动作协议，游戏运行时单独安装。

## 结果

LIBERO，40 局（trial 0），xhigh，BF16，温度 0。

| 模型 | 训练数据 | 成功 | 状态 |
|---|---|---:|---|
| Qwen3.5-4B / 9B（未训练） | — | 0/40 | 已确认 |
| 4B 学生，verl | 388 条 | 15/40 | 已确认 |
| 9B 学生，verl | 388 条 | 22/40 | 已确认 |
| 9B 学生，slime | 388 条 | 26/40 | 已确认 |
| 4B 学生 v2，verl | 1250 条 | 21/40 | 已确认 |
| 9B 学生 v2，verl | 1250 条 | 26/40 | 已确认 |
| PEAR-27B（教师，FP8） | — | 29/40 | 已确认 |

PEAR-27B 三阶段思维链退火（A3）：xhigh 档每轮生成 token 3643 → 1131（−69%），成功 29/40 → 32/40（配对 3 / 0）；medium、low 档成功率与 PEAR-27B 持平。LIBERO-Plus（56 局）32 vs 35，LIBERO-Pro（40 局）21 vs 21，每轮 token 分别减少 23% 与 4%。逐档位结果见 [退火配方页](recipes/anneal/README.md)，OPD 见各配方页，完整记录见 [`docs/pear-elite.md`](docs/pear-elite.md)。

LIBERO 上的结果为初步验证，后续换用更不易过拟合的环境。

## 安装

环境与上游 PEAR 相同，在 GPU 节点上：

```bash
git clone https://github.com/Robotic-Agent/PEAR-eLite.git && cd PEAR-eLite
export WORK_DIR=<所有节点可见的目录>     # 模型、环境与运行输出放在这里
bash scripts/install.sh libero_run       # 模型、LIBERO 与训练环境
```

verl 补丁随 `bash 3rdparty/verl/scripts/checkout_verl.sh` 应用（上游 43 个补丁之外另有 3 个，含 SDPA 下 Ulysses 序列并行的修复）。slime 环境单独构建，见 [`env/slime/README.md`](env/slime/README.md)。

## 评测

```bash
WORK_DIR=... bash tools/eval/run_eval.sh <模型目录> libero_all.json <输出名>
python tools/analysis/compare_paired.py <标签> <新 run 目录> <基线 run 目录>   # 按 episode 配对比较
```

## 数据

我们提供 PEAR-27B 在 LIBERO 系列上的 rollout 轨迹，每条包含每一轮的完整思维链、动作、机器人状态、两路相机画面与回合录像，可直接用于 SFT、蒸馏或离线分析。数据集公开在 ModelScope，可匿名下载：[candlest/pear-libero-trajectories](https://www.modelscope.cn/datasets/candlest/pear-libero-trajectories)

```bash
modelscope download --dataset candlest/pear-libero-trajectories --local_dir ./pear-libero-trajectories
```

| 目录 | 内容 | 状态 |
|---|---|---|
| `eval/` | 各模型的 LIBERO 评测轨迹与录像（每个 run 的帧与录像打包为 `episodes.tar`） | 已上传 |
| `rft_sampling/` | 思维链缩短（RFT）的采样轨迹 | 已上传 |
| `sft_data/` | 学生 SFT 数据（parquet） | 已上传 |
| `opd/` | OPD 训练日志与 rollout | 已上传 |
| `rollouts/` | PEAR-27B rollout 数据集，每回合一行 parquet（见下表） | 已上传 |

`rollouts/` 的规模：

| 子集 | 回合 | 成功 | 轮数 | 说明 |
|---|---:|---:|---:|---|
| `libero` | 1873 | 1337 | 25430 | LIBERO 4 个 suite |
| `libero_plus` | 3440 | 2074 | 48584 | 采集未完成 |
| `libero_pro` | 1599 | 1043 | 22292 | 采集未完成 |
| 合计 | 6912 | 4454 | 96306 | |

LIBERO-Plus 与 LIBERO-Pro 只作泛化测试，不进学生训练。两者的剩余分片后续补采，补采后在同一数据集中更新。打包脚本为 `tools/data/build_rollout_release.py`。

### 感知 VQA

同一数据集的 `perception_vqa/` 目录另提供两份机器人感知与任务进度的 VQA 数据。标签只来自人工或数据集自带的标注与画面中可见的内容，不使用仿真器的隐藏状态。

| 数据 | 规模 | 来源 | 题型 | 状态 |
|---|---|---|---|---|
| `roboatom` | 2481 行，74098 题（train 1911 / dev 339 / test 231 行，按任务划分，三者任务不重叠） | [RoboAtom](https://huggingface.co/datasets/InternRobotics/RoboAtom) 的原子动作标注，视频取自 LIBERO、RoboTwin 2.0、RoboDojo、RoboCerebra 与 RoboAtom-Bench | 片段是否完成、当前动作、当前技能（四选一）、多帧历史进度（已完成的子步骤与下一步）、任务是否完成 | 待上传 |
| `simvqa` | 1 万行 | LIBERO 仿真画面 | 当前画面可见的物体与机器人状态 | 待上传 |

每行为同一来源的一组题目（`roboatom` 每行至多 32 题、48 张图），图像为 JPEG，长边不超过 448。是非题在每个划分与来源内正负各半。

这两份数据尚未混入本仓库的任何训练，对学生模型闭环成功率的作用结论待定。作为参考，Qwen3-VL-2B-Instruct 零样本在 `roboatom` 抽检（每来源每题型 25 题）上的准确率：片段是否完成 0.576，任务是否完成 0.536，当前技能 0.440。

`roboatom` 中的视频帧来自上述公开数据集，使用时须遵守各数据集的许可。

## 目录

| 路径 | 内容 |
|---|---|
| `recipes/` | 四类配方的说明与入口索引 |
| `examples/models/` | verl 配方：学生 SFT 与 OPD（`qwen3_5_4b_pear/`）、27B LoRA 退火（`qwen3_8_27b_pear/train/`） |
| `examples/slime/` | slime 配方：9B SFT 与 ReOPD |
| `pear_slime/` | PEAR 轨迹到 slime 的数据转换、多轮 rollout 与教师打分 |
| `env/slime/` | slime 环境的构建与打包 |
| `scripts/` | 轨迹 → SFT 数据（`prepare_effort_sft.py` 等）、安装脚本 |
| `tools/` | 不依赖 Ray 的 LIBERO 评测、配对比较、NVFP4 教师量化、LoRA 合并、数据集打包 |
| `pear/` | 上游 PEAR：agent loop、环境、训练器；`pear/envs/games/` 包含《杀戮尖塔 2》和 Minecraft 适配器 |
| `3rdparty/verl/` | verl 固定版本与补丁 |

## 与上游的关系

本仓库基于 [mit-han-lab/pear](https://github.com/mit-han-lab/pear) 提交 `f32aa1f`，保留其完整历史与目录结构，便于合并上游更新。上游的训练、部署与机器人说明见 [`docs/README.md`](docs/README.md)。许可证与上游相同（Apache-2.0，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)）。

上游工作：[Training Reasoning Models in Simulation to Natively Use Real Robots](https://mit-han-lab.github.io/pear-blog/)。
