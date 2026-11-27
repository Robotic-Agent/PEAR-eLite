# 游戏环境

两套环境复用 PEAR 与 LIBERO 相同的 `StatefulEnv`、`EnvResult`、episode 配置和 JSON 动作通道，但有各自的观测、动作和提示词。游戏运行时单独安装，不影响原有机器人环境。

| 环境 | 后端与任务 | 观测 | 成功条件 | PEAR 配置 |
|---|---|---|---|---|
| 杀戮尖塔 2 | `sts2-rl` 使用的 NativeAOT `Sts2RunEnv`，铁甲战士整局模拟 | 当前界面与可用动作编号，纯文本 | 模拟器报告胜利 | [episode](sts2/configs/smoke.json) · [tool](sts2/configs/tool_config.yaml) |
| Minecraft | Minecraft-GYM-rs 的 Horizon + Pumpkin + Rust Step 协调器，`development-moving-sheep-v2` | 离屏第一人称 RGB、步数 | 服务端确认击中移动红羊 | [episode](minecraft/configs/smoke.json) · [tool](minecraft/configs/tool_config.yaml) |

每局按 seed 重置，一次 JSON 动作执行一个原生步骤。成功终止奖励 `1`；失败、超时和中间步骤不计成功。模拟器的 shaping reward 不进入 PEAR 的成功判定。

## 杀戮尖塔 2

沿用 [Candlest/sts2-rl](https://github.com/Candlest/sts2-rl) 的引擎，不搬运开悟训练进程、Steam 游戏、游戏资源或本地实验产物。适配器对接 [Zamiell/slay-the-spire-2-emulator](https://github.com/Zamiell/slay-the-spire-2-emulator) 的 `04cfe6df800156e5339d565e03236302363c829c`（Native API 10 / Run API 8）。支持的角色、楼层和事件以该模拟器为准，不等同于商业游戏的完整内容；观测中的部分事件保留数字 ID。

从本仓库根目录执行（Linux x86-64，需 .NET 9 SDK、clang、GNU binutils、zlib 开发包和 Python）：

```bash
git clone https://github.com/Zamiell/slay-the-spire-2-emulator.git /path/to/sts2-emulator
git -C /path/to/sts2-emulator checkout 04cfe6df800156e5339d565e03236302363c829c
git clone https://github.com/Candlest/sts2-rl.git /path/to/sts2-rl
git -C /path/to/sts2-rl checkout 0c7e5e4bd5f2dfa7963e924782c42a9f24f34374
bash pear/envs/games/sts2/build-linux.sh /path/to/sts2-emulator /path/to/sts2-rl

python -m venv /path/to/sts2-venv
export PEAR_STS2_PYTHON=/path/to/sts2-venv/bin/python
"$PEAR_STS2_PYTHON" -m pip install gymnasium numpy pytest
export PYTHONPATH=/path/to/sts2-emulator/src
PEAR_TEST_NATIVE_STS2=1 "$PEAR_STS2_PYTHON" -m pytest -q tests/sim/test_sts2_game_in_sim.py
```

Linux 构建脚本只修改外部模拟器 checkout，复用 `sts2-rl` 的 ELF 导出修复；不编译开悟训练栈。Windows 可在未打 Linux 补丁的新 checkout 中直接运行上游 `scripts/build.sh win-x64`（Git Bash + MSVC Build Tools），venv 的 Python 则用 `Scripts/python.exe`。已分别用 Windows 和 WSL Ubuntu 新编译的引擎验证固定种子、地图、战斗及超时；WSL 还通过了 PEAR `SimulatorProcess` 收发测试。

## Minecraft

复用 [Candlest/Minecraft-GYM-rs](https://github.com/Candlest/Minecraft-GYM-rs) `81cf706389f9c0ad7fdf37a12ab697b598552b7a` 的代码和构建清单（私有仓库，需要读取权限）。原项目固定三项上游：**Pumpkin**（服务端）、**Horizon**（客户端）和 **GymCraft**（原始环境语义与 Java 对照）。这里接入的是 Horizon + Pumpkin 的真实移动羊场景，不是 GymCraft 回归场景，也不使用 `SyntheticAim`。

### Linux x86-64：构建与资源准备

需要 Rustup/Cargo 1.97.0、Python 3.11+ 和 JDK 25（只用于准备资源）。Ubuntu 24.04 上的构建和运行依赖：

```bash
sudo apt-get install build-essential clang cmake pkg-config libssl-dev libasound2-dev \
  libudev-dev libx11-dev libxi-dev libxrandr-dev libxcursor-dev libxinerama-dev \
  libwayland-dev libxkbcommon-dev libxkbcommon-x11-0 libvulkan1 xvfb xauth curl unzip python3-venv
rustup toolchain install 1.97.0

git clone https://github.com/Candlest/Minecraft-GYM-rs.git
cd Minecraft-GYM-rs
git checkout --detach 81cf706389f9c0ad7fdf37a12ab697b598552b7a
export GAMEBENCH_ROOT="$PWD"
python3 -m venv .venv
export PEAR_MINECRAFT_PYTHON="$PWD/.venv/bin/python"
"$PEAR_MINECRAFT_PYTHON" -m pip install -e '.[rust]'

mkdir -p runs upstream
for engine in horizon pumpkin; do
  url=$("$PEAR_MINECRAFT_PYTHON" -c \
    'import json, sys; print(json.load(open("runtime.experimental.lock.json"))[sys.argv[1]]["archive_url"])' "$engine")
  curl -fL --retry 3 -o "runs/$engine-source.zip" "$url"
done
# The build scripts verify both archives against runtime.experimental.lock.json.
"$PEAR_MINECRAFT_PYTHON" scripts/build_horizon_bridge.py --mode build

# The unmodified Horizon tree supplies the resource preparation scripts and research lock.
unzip -q runs/horizon-source.zip -d upstream
horizon="$PWD/upstream/horizon-441f49d57e344bd56b90fd4af5c4e443ff3937c9"
inputs="$PWD/upstream/minecraft-26.3-inputs"
"$PEAR_MINECRAFT_PYTHON" scripts/fetch_horizon_inputs.py
PUMPKIN_MINECRAFT_SERVER_JAR="$inputs/server.jar" \
  "$PEAR_MINECRAFT_PYTHON" scripts/build_pumpkin_bridge.py --mode build --builtin \
  --workspace upstream/pumpkin-gamebench-builtin
cargo +1.97.0 build --manifest-path rust/Cargo.toml --locked --release -p gamebench-step-coordinator
JAVA_HOME=/path/to/jdk-25 HORIZON_PREPARE_JOBS=2 \
  "$horizon/target/fast/horizon-prepare" --prepare --root "$horizon" \
  --minecraft-dir "$inputs" --server-jar "$inputs/server.jar"
```

若已有同版本准备好的资源，可跳过 `horizon-prepare`，在启动时传 `--resources /path/to/horizon-client/26.3`。Pumpkin 构建仍需要 `server.jar` 生成 datapack；上面指定校验过的本地 jar，避免 Cargo 构建时再次联网下载。Windows 构建见[原项目 M2 说明](https://github.com/Candlest/Minecraft-GYM-rs/blob/81cf706389f9c0ad7fdf37a12ab697b598552b7a/docs/horizon_pumpkin_m2.md)。

### 启动与验收

从 PEAR-eLite 根目录运行。Horizon 在隐藏窗口中渲染到纹理，Linux 无桌面节点用 Xvfb 提供窗口系统，不能把 `--offscreen` 换成没有 renderer 的 `--headless`。Windows 直接调用 Python，不加 `env -u WAYLAND_DISPLAY xvfb-run -a`，并传对应的 `python.exe` 和 `gamebench-step-coordinator.exe` 路径。

```bash
env -u WAYLAND_DISPLAY xvfb-run -a "$PEAR_MINECRAFT_PYTHON" -m pear.envs.games.minecraft.launch \
  --gamebench-root "$GAMEBENCH_ROOT" \
  --coordinator "$GAMEBENCH_ROOT/rust/target/release/gamebench-step-coordinator" \
  --output /path/to/new-arena-output --port 25585 --simulation-distance 5 \
  --width 960 --height 540 --require-hardware-renderer --smoke
```

- `--output` 必须是新目录。`--smoke` 用真实 RGB 完成一次服务端确认的命中，再跑一次单 tick 超时，最后关闭客户端和服务端。`report.json` 记录实际帧尺寸、硬件渲染器检查结果和退出码，具体适配器型号见 `horizon.log`。
- 硬件验收要提供可用的 Vulkan GPU 驱动并保留 `--require-hardware-renderer`。如果只验收功能，可安装 `mesa-vulkan-drivers`，用 `VK_DRIVER_FILES` 指向本机的 `lvp_icd*.json`，去掉 `--require-hardware-renderer`；软件渲染的结果不算 GPU 验收。
- 不加 `--smoke` 时保留专用 arena，并打印 `PEAR_MINECRAFT_RUNTIME=/.../pear-arena.json`。在运行 PEAR 的同一节点设置该变量。评测结束后用 Ctrl+C 或向 Python 启动器发送 SIGTERM；端点仅绑定本机，勿共享运行时目录。
- 一个 arena 同时只能跑一局。Step 通信失败会丢弃当前回合，不重试可能已经执行的动作。策略看不到服务端坐标、实体 ID 或隐藏轨迹。该场景仍是单任务开发环境，不作为正式 benchmark。

保持 arena 打开后，可用以下命令验收 Linux 上的 `SimulatorProcess` 和单节点 Ray 传输：

```bash
"$PEAR_MINECRAFT_PYTHON" -m pip install pytest 'ray==2.58.0'  # 与 PEAR 的当前依赖锁一致
export PEAR_MINECRAFT_RUNTIME=/path/to/new-arena-output/pear-arena.json
export no_proxy="127.0.0.1,127.0.0.2,localhost${no_proxy:+,$no_proxy}"
export NO_PROXY="$no_proxy"
"$PEAR_MINECRAFT_PYTHON" -m pytest -q tests/sim/test_minecraft_game_in_sim.py
```

已在 WSL Ubuntu 24.04 原生构建 Horizon、Pumpkin 和 Step 协调器。复用已准备好的 26.3 资源，在 Xvfb + Mesa llvmpipe（Vulkan 软件渲染）实测 640×360 RGB、第一步服务端确认命中、第二局单 tick 超时，两个引擎退出码均为 0。PEAR `SimulatorProcess` 和 Ray 2.58.0 单节点的开局、动作与回合结束测试均通过。Windows 的硬件 Vulkan 验收在 RTX 4060 上通过；Linux 硬件 Vulkan 和完整模型 rollout 尚未验证。

## 接入 PEAR agent loop

设 `data.train_files` 或 `data.val_files` 为对应的 episode JSON，`actor_rollout_ref.rollout.multi_turn.tool_config_path` 为对应的 `tool_config.yaml`，并设置 `PEAR_LOG_DIR`。`env_class` 由 `pkgutil.resolve_name` 解析；配置中的 `data_source` 指向 `env` 的一个字段，`env` 的每个字段都是候选值列表或整数范围。

`SimulatorProcess` 使用 Linux 进程接口。请在运行时所在的 Ray 节点设置上述 Python 和路径变量。Minecraft 只在已启动 arena 的节点声明一个自定义资源 `{"minecraft": 1}`，PEAR 会把唯一的 Minecraft 模拟进程放在该节点。Minecraft 的 PEAR agent loop 进程也需要 Pillow 来反序列化帧。现有 `tools/eval/run_eval.sh` 是 LIBERO 专用入口；游戏使用上述 agent loop 配置或各自的本地冒烟入口。
