## SFT

用 PEAR-27B 的成功轨迹对 Qwen3.5-4B / 9B 学生做全参数微调。仓库提供 verl 配方和 slime 的 9B 配方。首轮使用 388 条成功轨迹，项目文章报告的 4B / 9B SFT 成绩使用 1250 条。

[查看 SFT 配方与入口](https://github.com/Robotic-Agent/PEAR-eLite/tree/main/recipes/sft)

## OPD

回放式在线蒸馏（ReOPD）：历史上下文来自教师轨迹，学生用当前策略续写当前轮，教师逐 token 打分；不在环境里跑完整局，也不使用任务奖励。当前配方使用 slime，思考档位按 `xhigh → medium → low` 排列。4B 正式训练还在跑，先验证 on-policy 更新能否稳定迭代，再通过同口径闭环评测判断相对 SFT 的收益。

[查看 OPD 配方与入口](https://github.com/Robotic-Agent/PEAR-eLite/tree/main/recipes/opd)

## Anneal

在 PEAR-27B 上做三阶段思维链退火。每一档位先采样，再为每个任务挑最多 4 条生成 token 最少的成功轨迹做 LoRA SFT，合并后进入下一档位。三个阶段分别选出 135、48、74 条；单阶段 RFT 的评测不等同于三阶段退火结果。

[查看三阶段退火配方](https://github.com/Robotic-Agent/PEAR-eLite/tree/main/recipes/anneal)

## Slime

将 PEAR 的多轮多模态轨迹转换为 slime 训练或蒸馏所需的输入，提供固定版本的 slime、SGLang 和 Megatron 环境。

[查看 slime 适配说明](https://github.com/Robotic-Agent/PEAR-eLite/tree/main/recipes/slime)
