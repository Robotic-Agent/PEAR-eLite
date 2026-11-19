## 学生模型：LIBERO

LIBERO `trial 0`，四个 suite 各 10 局，均使用 `xhigh` 档。学生 BF16，教师 FP8。SFT v2 的 1250 条成功轨迹是已上传 6912 局的子集，来自原版 LIBERO 的 `xhigh` 采样，排除了这些评测回合。

| 模型 | SFT 训练数据 | LIBERO 成功局数 |
| --- | ---: | ---: |
| PEAR-27B（教师） | — | 29 / 40 |
| Qwen3.5-9B-PEAR v2（verl） | 1250 条 | 26 / 40 |
| Qwen3.5-4B-PEAR v2（verl） | 1250 条 | 21 / 40 |
| Qwen3.5-4B / 9B（未训练） | — | 0 / 40 |

同用 verl 的 9B，在 388 条轨迹上训练时是 22/40。与 v2 逐局配对，只有 v2 成功的有 7 局，只有原版本成功的有 3 局（McNemar $p = 0.34$）。另一个 slime 版本用 388 条轨迹也跑到 26/40，不混入这组同后端比较。4B v2 在 `libero_10` 子集是 0/10，整组评测中每轮平均生成 4907 token。

## ReOPD：训练链路与闭环效果

ReOPD 用教师轨迹提供固定的历史上下文，学生用当前策略生成当前轮，教师返回逐 token 分数。它验证的是**单轮生成上的 on-policy 更新**，不做完整环境 rollout，也不使用任务奖励。

- **可用性**：已有试跑完成了采样、教师打分和参数更新，4B 正式训练在 slime 上进行。早期试跑的训练指标只用来检查链路。
- **有效性**：还没有成套的闭环评测结果。正式训练结束后，在相同任务、回合和思考档位下与 SFT 起点比较；SFT 的 21/40 不是 ReOPD 的分数。

## 三阶段退火：LIBERO

这里比较的是 27B 教师和做完 A1 → A2 → A3 的 27B 模型。每格的成功局数都以 40 局为分母；token 统计是每轮生成量，不是整局开销或实测时延。教师的 `xhigh` 结果用 FP8，表内其余结果用 BF16。

| 思考档位 | 原模型 → A3 每轮 token | 原模型 → A3 成功局数 |
| --- | ---: | ---: |
| `xhigh` | 3643 → 1131（约 −69%） | 29 → 32 |
| `medium` | 947 → 597（约 −37%） | 11 → 12 |
| `low` | 1029 → 647（约 −37%） | 12 → 10 |

`low` 档只做 A1 时是 4/40，A2 为 13/40，A3 为 10/40。完整逐阶段结果见[退火配方](https://github.com/Robotic-Agent/PEAR-eLite/blob/main/recipes/anneal/README.md)。

## 扰动任务：LIBERO-Plus / Pro

两组扰动任务均未用于训练。评测用 `xhigh` 档，按扰动类型分层抽取，在相同回合上比较 A3 和原模型。

| 环境 | 原模型成功局数 | A3 成功局数 | 原模型 → A3 每轮 token |
| --- | ---: | ---: | ---: |
| LIBERO-Plus | 35 / 56 | 32 / 56 | 2155 → 1656（约 −23%） |
| LIBERO-Pro | 21 / 40 | 21 / 40 | 1987 → 1904（约 −4%） |

任务分布改变后，思维链缩短幅度明显减小。表中数值是这批评测的观察结果，不是对成功率等价性的检验。

## 代码与轨迹

在仓库根目录运行评测前，先按仓库 README 安装环境，再把 `WORK_DIR` 和模型目录替换为实际路径：

```bash
WORK_DIR=/path/to/work bash tools/eval/run_eval.sh /path/to/model libero_all.json student-eval
```

完整的 6912 局教师 rollout 已上传到 [ModelScope 数据集](https://www.modelscope.cn/datasets/candlest/pear-libero-trajectories) 的 `rollouts/`。仓库里的[实验记录](https://github.com/Robotic-Agent/PEAR-eLite/blob/main/docs/pear-elite.md)还包括序列并行问题和早期单阶段 RFT；它们与本页的三阶段退火分开记录。代码仓库待公开。
