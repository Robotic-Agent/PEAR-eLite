# OPD：回放式在线蒸馏（ReOPD）

prompt 为 PEAR-27B 一条已记录轨迹截到第 t 个用户轮为止（系统、用户 0、助手 0、……、用户 t），学生生成第 t 轮，教师 SGLang 服务对学生的整段序列返回逐 token log-prob。loss 为 slime 的 OPD：advantage = −kl_coef ×（log p_学生 − log p_教师），没有任务奖励。

| 文件 | 作用 |
|---|---|
| `pear_slime/opd_data.py` | `pear_slime/convert_sft_data.py` 转换后的 parquet → OPD prompt，每条轨迹随机截 K 轮 |
| `pear_slime/opd_generate.py` | 按评测时的格式渲染 prompt；思考达到上限仍未结束时补 `</think>`（不计 loss），再在 `PEAR_OPD_SOLUTION_BUDGET` 内写动作 |
| `pear_slime/opd_teacher.py` | 教师打分，核对教师与学生的 token 一致（图像占位位置返回 0）；不一致的样本从 loss 中去掉，超过 10% 时停止 |
| `examples/slime/qwen3_5_9b_pear/run_opd.sh` | 单节点 8 × H100：教师 2 卡，学生 TP 2 × CP 3；`STAGES` 默认 `xhigh:40:32768 medium:40:8192 low:40:8192` |

```bash
source <slime 环境>/activate.sh
python -m pear_slime.convert_sft_data <SFT parquet> sft_slime.parquet
python -m pear_slime.opd_data sft_slime.parquet opd_prompts.parquet --turns-per-episode 4
STUDENT=<学生 HF 目录> TEACHER=<PEAR-27B 目录> TRAIN_DATA=opd_prompts.parquet RUN=<输出目录> \
  bash examples/slime/qwen3_5_9b_pear/run_opd.sh
```

## 已知问题

- medium / low 阶段的历史轮来自 27B 的 xhigh 轨迹，测试时历史轮是学生自己的低档位输出，结论待定。
- 强制结束思考的样本出现过 3 个 token 的长度差，原因未查明，已从 loss 中去掉。
- 训练期间不做评测：每阶段结束后用 HF checkpoint 在对应档位评测 40 局。
