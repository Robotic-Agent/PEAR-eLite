# slime 适配

| 组成 | 位置 | 说明 |
|---|---|---|
| 环境 | `env/slime/` | slime / SGLang / Megatron 版本固定，自带 Python 与 CUDA 13 工具链，可打包为目录 tar 或 Docker 镜像 |
| PEAR 多轮多模态适配 | `pear_slime/` | `episode.py`、`sft_rollout.py`：一条多轮多图轨迹作为一个样本，按推理时的拼接渲染，助手轮计 loss；`convert_sft_data.py`：verl SFT parquet → slime prompt 数据 |
| 9B SFT | `examples/slime/qwen3_5_9b_pear/run_sft.sh` | 8 × H100 |
| 9B OPD | `examples/slime/qwen3_5_9b_pear/run_opd.sh` | 见 [OPD](../opd/README.md) |

## 已知限制

- RTX 5090 上不可用：Transformer Engine 对 head dim 256 没有可用的注意力后端（cuDNN fused 拒绝该输入，FlashAttention 4 报 `CUDADialectError`），slime 配方只在 H100 上运行。
- Megatron 要求 `--seq-length` 为 2 × CP 的倍数（slime 训练变长打包序列，该值只是名义值）。
