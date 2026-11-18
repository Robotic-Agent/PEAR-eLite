# slime 环境

slime / SGLang / Megatron 训练与推理环境，版本全部固定，安装到 `$PREFIX`（默认 `/opt/slime`）。环境自带 Python、uv 和 CUDA 13 编译工具链（来自 nvidia wheel），不依赖宿主机的 CUDA 和 Python。版本对齐 slime `docker/Dockerfile`（commit `79aa41a8`，SGLang v0.5.21 线），CUDA 扩展编译架构为 9.0 / 10.0 / 12.0（H100、B200、RTX 5090）。slime 官方镜像只编译 9.0 / 10.0 / 10.3，不能在 5090 上运行。

## 文件

| 文件 | 内容 |
|---|---|
| `versions.env` | 全部版本与 commit |
| `fetch_sources.sh <目录>` | 下载 GitHub 上的源码包和 wheel，生成 `SHA256SUMS`。在能访问 GitHub 的机器上运行 |
| `build.sh <源码目录> [python 目录]` | 构建环境。每步完成后写标记，重跑时从未完成的步骤继续 |
| `check.py` | 单卡检查：版本、GPU 架构、各 CUDA 扩展的一次前向，全部通过时输出 `CHECK_OK` |
| `pack.sh <tar>` | 把 `$PREFIX` 打成一个 tar，排除下载缓存、构建标记和 Rust 工具链 |
| `Dockerfile` | 用同一个 `build.sh` 构建镜像（Ubuntu 24.04） |

## 主要版本

| 组件 | 版本 |
|---|---|
| Python | 3.12.15 |
| torch / torchvision / torchaudio | 2.13.0 / 0.28.0 / 2.11.0（cu130） |
| CUDA 编译器 | 13.0.88（与 torch 的 CUDA 运行时一致） |
| SGLang / sglang-kernel / flashinfer | 0.5.21（含 slime 补丁）/ 0.4.7 / 0.6.18 |
| transformers | 5.12.1 |
| Transformer Engine | 2.18.0 |
| Megatron-LM | `ae76c60e`（含 slime 补丁） |
| apex / torch_memory_saver / mamba | `10417ace` / `bdab8c55` / `0048fbf2` |
| causal-conv1d / flash-linear-attention / tilelang | 1.5.3.post1 / 0.5.2 / 0.1.12 |
| NCCL | 2.30.7 |

完整依赖列表在构建后的 `$PREFIX/LOCK.txt`。

## 使用已打包的环境

环境中的路径是绝对路径，必须解压到相同的 `$PREFIX`：

```bash
tar -C / -xf slime-env.tar          # 得到 /opt/slime
source /opt/slime/activate.sh       # CUDA_HOME、PATH、PYTHONPATH（Megatron-LM）、LD_LIBRARY_PATH
python env/slime/check.py           # 可选：在 pear-lite 仓库目录下运行单卡检查
```

运行要求：x86_64，glibc ≥ 2.35，NVIDIA 驱动 ≥ 580（CUDA 13.0）。宿主机不需要安装 CUDA。

集群上的用法：tar 放在共享存储，任务启动后解压到容器本地 `/opt/slime`，不要直接在网络文件系统上运行（小文件读取慢）。

## 从源码构建

```bash
bash env/slime/fetch_sources.sh /path/to/slime-sources        # 需要 GitHub
bash env/slime/build.sh /path/to/slime-sources                # 需要 PyPI、PyTorch 索引、Rust 工具链服务器、crates.io
source /opt/slime/activate.sh && python env/slime/check.py
bash env/slime/pack.sh /path/to/slime-env.tar
```

构建机需要 `gcc`/`g++`、`git`、`curl`。第二个参数可以给一个已解压的 python-build-standalone 目录；不给时由 uv 下载 Python。

可调变量：

| 变量 | 默认值 | 作用 |
|---|---|---|
| `PREFIX` | `/opt/slime` | 安装位置 |
| `JOBS` | 32 | 编译并行数 |
| `CUDA_ARCHS` / `NVTE_CUDA_ARCHS` | `9.0;10.0;12.0` / `90;100f;120f` | 编译架构，A100 加 `8.0` / `80` |
| `PYPI_INDEX_URL` | PyPI | Python 包镜像 |
| `TORCH_FIND_LINKS` | 空（使用 download.pytorch.org） | torch cu130 wheel 镜像目录 |
| `RUSTUP_DIST_SERVER` / `RUSTUP_UPDATE_ROOT` | static.rust-lang.org | Rust 工具链镜像 |
| `CARGO_REGISTRY_MIRROR` | crates.io | crates 镜像 |

国内网络（商汤开发机实测，pypi.org 约 18 KB/s，镜像约 11 MB/s）：

```bash
PYPI_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
TORCH_FIND_LINKS=https://mirrors.aliyun.com/pytorch-wheels/cu130/ \
RUSTUP_DIST_SERVER=https://rsproxy.cn RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup \
CARGO_REGISTRY_MIRROR=sparse+https://rsproxy.cn/index/ \
bash env/slime/build.sh /path/to/slime-sources
```

在 128 核的开发机上（JOBS=32），CUDA 扩展编译约 15 分钟：Transformer Engine 使用预编译 wheel（约 1 分钟），apex 约 6 分钟，causal-conv1d + mamba 约 7 分钟。

## Docker

```bash
docker build -f env/slime/Dockerfile --build-context sources=/path/to/slime-sources \
  [--build-arg PYPI_INDEX_URL=... 等，同上表] -t pear-slime:0.5.21 .
```

镜像基于 Ubuntu 24.04，另装 RDMA 用户态库（`libibverbs1`、`librdmacm1`），供 NCCL 走 InfiniBand / RoCE。Dockerfile 尚未在本集群构建验证（开发机无 Docker Hub 访问）。

## 与 slime 官方镜像的差异

| 项目 | 官方镜像 | 本环境 |
|---|---|---|
| CUDA 扩展架构 | 9.0、10.0、10.3 | 9.0、10.0、12.0（可配置） |
| CUDA 来源 | 基础镜像的系统 CUDA | 环境内 nvidia wheel，`CUDA_HOME` 指向 `site-packages/nvidia/cu13` |
| CUDA 13 编译补丁 | GB10 说明中手工处理 | `build.sh` 自动补 NVTX 3 头文件与 `cuda_profiler_api.h` |
| 部署方式 | Docker 镜像 | 目录 tar 或 Docker 镜像 |

## 已知限制

- `nvidia-resiliency-ext` 0.7.0（Megatron 异步保存检查点）只有 glibc 2.39 的 wheel。构建时按 glibc 2.39 平台安装；在 glibc 2.35（Ubuntu 22.04）主机上该包不能加载，异步保存不可用，其余功能不受影响。NGC PyTorch 25.06 镜像为 Ubuntu 24.04。
- `check.py` 只做单卡检查。多卡 NCCL、SGLang 服务启动、slime 训练循环由各任务的 smoke 验证。
