# PEAR-eLite 配方索引

四类配方的入口、数据与当前状态。脚本保留在原位置（`examples/`、`scripts/`、`pear_slime/`、<internal job scripts>），本目录只做索引，避免改动上游 PEAR 的路径。实测结果见 [`docs/pear-elite.md`](../docs/pear-elite.md)。

| 配方 | 说明 | 后端 | 硬件 | 状态 |
|---|---|---|---|---|
| [SFT](sft/README.md) | PEAR-27B 轨迹 → Qwen3.5-4B/9B 学生全参数 SFT | verl、slime | 8 × RTX 5090（verl）、8 × H100（slime） | 已完成训练；4B v2 评测结论待定 |
| [OPD](opd/README.md) | PEAR-27B 教师 → 9B 学生的回放式在线蒸馏（ReOPD），每轮思考档位课程 | slime | 8 × H100 | 运行中 |
| [退火](anneal/README.md) | PEAR-27B 三阶段思维链退火：xhigh → medium → low | verl（LoRA） | 8 × RTX 5090 | A1 已完成，A2、A3 运行中 |
| [slime](slime/README.md) | slime / SGLang / Megatron 环境与 PEAR 多轮多模态适配 | slime | H100（5090 不可用） | 已完成 |

游戏强化学习环境放在 [`pear/envs/games/`](../pear/envs/games/README.md)。
