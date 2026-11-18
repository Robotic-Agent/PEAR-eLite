# 游戏强化学习环境

本目录用于放置游戏环境，与 `pear/envs/libero/` 等机器人环境使用同一套接口，可直接接入 PEAR 的 agent loop、rollout 与评测。目前没有已实现的环境。

## 一个环境的组成

```
pear/envs/games/<game>/
  __init__.py
  env.py                    # 环境类
  environment_prompt.md     # 拼入系统提示的环境说明（观测、动作、规则）
  configs/                  # 数据源与 episode 配置（yaml）
```

## 环境类

继承 `pear.sim.env` 中的一种基类：

| 基类 | 适用 | 需要实现 |
|---|---|---|
| `StatelessEnv` | 进程不保存回合状态：每次调用由 env args 重建回合、重放已有的原生动作、再执行一步。适合确定性、重建便宜的游戏 | `rollout(history, action) -> EnvResult`，`history` 为已执行的原生动作，`action` 为 `None` 时返回初始观测；类属性 `ROLLOUT_TIMEOUT_S` |
| `StatefulEnv` | 进程从打开到关闭保存一个回合；回合无法继续时抛出 `EpisodeLost`。适合重建昂贵或不确定的游戏 | `open() -> EnvResult`（初始观测）、`step(action) -> EnvResult`、`close()` |

两类环境都必须实现类方法 `setup_env`（启动模拟进程）与 `task_name`（由 env args 得到任务名），并给出 `ENVIRONMENT_PROMPT`；`pear/envs/libero/env.py` 另有 `server_n`（每节点的模拟进程数），可作参考。每次调用返回 `pear.sim.schemas.EnvResult`：相机帧 `images`、状态 `state`、执行的原生动作 `actions`、每个原生动作的 `rewards`、`done`、`executed`、`outcome`、`instruction`。

## 接入

在 episode 配置中写环境类的完整路径，agent loop 用 `pkgutil.resolve_name` 加载：

```json
{"env_class": "pear.envs.games.<game>.env.<GameEnv>", "data_source": "...", "env": {"...": "..."}}
```
