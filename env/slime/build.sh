#!/usr/bin/env bash
# The slime / SGLang / Megatron environment at $PREFIX (default /opt/slime), self-contained: its own Python, uv and
# CUDA 13 toolkit (from wheels), the CUDA extensions built for CUDA_ARCHS. Needs PyPI and download.pytorch.org; the
# GitHub inputs come from fetch_sources.sh. Every step leaves a stamp, so a rerun resumes after the last finished step.
#   bash env/slime/build.sh <sources dir> [python-build-standalone install dir]
# Where PyPI or download.pytorch.org are slow, point both at mirrors (same wheels, the versions stay pinned):
#   PYPI_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
#   TORCH_FIND_LINKS=https://mirrors.aliyun.com/pytorch-wheels/cu130/ \
#   RUSTUP_DIST_SERVER=https://rsproxy.cn RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup \
#   CARGO_REGISTRY_MIRROR=sparse+https://rsproxy.cn/index/ bash env/slime/build.sh ...
# Then: source $PREFIX/activate.sh; python env/slime/check.py; bash env/slime/pack.sh <out tar>
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd -P)
# shellcheck source=versions.env
source "$HERE/versions.env"
SRC_IN=$(cd "${1:?sources dir from fetch_sources.sh}" && pwd -P)
PY_IN=${2:-}
JOBS=${JOBS:-32}

mkdir -p "$PREFIX"/{bin,src,stamps}
export PATH=$PREFIX/bin:$PATH
export UV_CACHE_DIR=${UV_CACHE_DIR:-$PREFIX/.cache/uv} UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
export UV_HTTP_TIMEOUT=300
[ -z "${PYPI_INDEX_URL:-}" ] || export UV_DEFAULT_INDEX=$PYPI_INDEX_URL
TORCH_FIND_LINKS=${TORCH_FIND_LINKS:-}
# the Rust toolchain that builds SGLang's extensions; a build-time tool only, pack.sh leaves it out
export RUSTUP_HOME=$PREFIX/rust/rustup CARGO_HOME=$PREFIX/rust/cargo
export PATH=$CARGO_HOME/bin:$PATH CARGO_BUILD_JOBS=$JOBS
ENV=$PREFIX/env
PY=$ENV/bin/python
CONSTRAINTS=$PREFIX/constraints.txt

step() {
  local name=$1
  shift
  if [ -f "$PREFIX/stamps/$name" ]; then
    echo "== $name: done before"
    return 0
  fi
  echo "== $name ($(date -u +%H:%M:%S))"
  "$@"
  touch "$PREFIX/stamps/$name"
}
uvpip() { uv pip install --python "$PY" "$@"; }
site() { "$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"; }

s_verify() { (cd "$SRC_IN" && sha256sum -c --quiet SHA256SUMS); }

s_uv() {
  tar -xzf "$SRC_IN/uv-x86_64-unknown-linux-gnu.tar.gz" -C "$PREFIX/bin" --strip-components=1
  uv --version
}

s_python() {
  rm -rf "$PREFIX/python"
  mkdir -p "$PREFIX/python"
  if [ -n "$PY_IN" ]; then
    cp -a "$PY_IN/." "$PREFIX/python/"
  else
    UV_PYTHON_DOWNLOADS=automatic uv python install --install-dir "$PREFIX/.cache/python" "$PYTHON_VERSION"
    cp -a "$PREFIX/.cache/python"/cpython-"$PYTHON_VERSION"-linux-x86_64-gnu/. "$PREFIX/python/"
  fi
  "$PREFIX/python/bin/python3" --version | grep -q "$PYTHON_VERSION"
}

unpack() {  # tarball prefix -> $PREFIX/src/<name>, as a git repository (the patches apply with git apply)
  local tarball=$1 name=$2
  rm -rf "$PREFIX/src/$name"
  mkdir -p "$PREFIX/src/$name"
  tar -xzf "$tarball" -C "$PREFIX/src/$name" --strip-components=1
  git -C "$PREFIX/src/$name" init -q
  git -C "$PREFIX/src/$name" -c user.name=build -c user.email=build@localhost add -A
  git -C "$PREFIX/src/$name" -c user.name=build -c user.email=build@localhost commit -qm "upstream $(basename "$tarball")"
}

s_sources() {
  unpack "$SRC_IN"/slime-"$SLIME_COMMIT".tar.gz slime
  unpack "$SRC_IN"/sglang-v"$SGLANG_VERSION".tar.gz sglang
  unpack "$SRC_IN"/Megatron-LM-"$MEGATRON_COMMIT".tar.gz Megatron-LM
  unpack "$SRC_IN"/apex-"$APEX_COMMIT".tar.gz apex
  unpack "$SRC_IN"/torch_memory_saver-"$TMS_COMMIT".tar.gz torch_memory_saver
  unpack "$SRC_IN"/mamba-"$MAMBA_COMMIT".tar.gz mamba
  unpack "$SRC_IN"/NVTX-"$NVTX_COMMIT".tar.gz NVTX
  # slime's patches to Megatron and SGLang, as its Dockerfile applies them
  local p=$PREFIX/src/slime/docker/patch/$SLIME_PATCH_VERSION patch
  for patch in megatron.patch megatron-sglang-aligned.patch; do
    if [ -f "$p/$patch" ]; then
      git -C "$PREFIX/src/Megatron-LM" apply --3way "$p/$patch"
      echo "Megatron-LM: $patch"
    fi
  done
  for patch in sglang.patch sglang-top_p.patch sglang-release_hicache.patch sglang-pull_weights.patch \
    sglang-deterministic.patch sglang-radio_layerscale.patch sglang-nemotron-multi-image.patch; do
    if [ -f "$p/$patch" ]; then
      git -C "$PREFIX/src/sglang" apply "$p/$patch"
      echo "sglang: $patch"
    fi
  done
  if git -C "$PREFIX/src/Megatron-LM" grep -qn '^<<<<<<< ' -- .; then
    echo "Megatron-LM patch conflicts"
    return 1
  fi
}

s_venv() {
  uv venv --clear --python "$PREFIX/python/bin/python3" "$ENV"
}

s_rust() {
  local dist=${RUSTUP_DIST_SERVER:-https://static.rust-lang.org}
  local root=${RUSTUP_UPDATE_ROOT:-$dist/rustup}
  mkdir -p "$PREFIX/.cache" "$CARGO_HOME"
  curl -fL --retry 5 -o "$PREFIX/.cache/rustup-init" "$root/dist/x86_64-unknown-linux-gnu/rustup-init"
  chmod +x "$PREFIX/.cache/rustup-init"
  RUSTUP_DIST_SERVER=$dist RUSTUP_UPDATE_ROOT=$root "$PREFIX/.cache/rustup-init" -y --no-modify-path \
    --profile minimal --default-toolchain "$RUST_VERSION" --component clippy,rustfmt
  if [ -n "${CARGO_REGISTRY_MIRROR:-}" ]; then
    cat > "$CARGO_HOME/config.toml" <<EOF
[source.crates-io]
replace-with = "mirror"

[source.mirror]
registry = "$CARGO_REGISTRY_MIRROR"
EOF
  fi
  cargo --version
}

# torch, SGLang (patched source) and their pins in one resolution, on PyTorch's CUDA 13 index
s_sglang() {
  # the source is a tarball without SGLang's git tags, so setuptools-scm gets the version from here
  export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=$SGLANG_VERSION
  if [ -n "$TORCH_FIND_LINKS" ]; then
    uvpip --find-links "$TORCH_FIND_LINKS" "torch==$TORCH_VERSION+$CUDA_TAG" "torchvision==$TORCHVISION_VERSION+$CUDA_TAG" \
      "torchaudio==$TORCHAUDIO_VERSION+$CUDA_TAG" -e "$PREFIX/src/sglang/python"
  else
    uvpip --torch-backend="$CUDA_TAG" "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
      "torchaudio==$TORCHAUDIO_VERSION" -e "$PREFIX/src/sglang/python"
  fi
  # the versions everything after must keep: slime's Dockerfile pins the first set, and every CUDA library wheel torch
  # brought stays at its version (nvjitlink is pinned below with the compiler, which it must match)
  "$PY" - > "$CONSTRAINTS" <<'EOF'
import importlib.metadata as m
names = {"torch", "torchvision", "torchaudio", "sglang-kernel", "flashinfer-python", "flashinfer-cubin",
         "transformers", "cuda-python", "cuda-bindings"}
for d in m.distributions():
    n = d.metadata["Name"].lower().replace("_", "-")
    if n.startswith(("nvidia-", "cuda-")):
        names.add(n)
names -= {"cuda-toolkit", "nvidia-nvjitlink", "nvidia-cuda-nvcc", "nvidia-cuda-crt", "nvidia-nvvm", "nvidia-cuda-cccl",
          "nvidia-cuda-profiler-api", "nvidia-cuda-nvdisasm"}
for n in sorted(names):
    try:
        print(f"{n}=={m.version(n)}")
    except m.PackageNotFoundError:
        pass
EOF
  cat >> "$CONSTRAINTS" <<EOF
pyarrow==$PYARROW_VERSION
cuda-toolkit==$CUDA_TOOLKIT_VERSION
nvidia-cuda-nvcc==$CUDA_NVCC_VERSION.*
nvidia-cuda-crt==$CUDA_NVCC_VERSION.*
nvidia-nvvm==$CUDA_NVCC_VERSION.*
nvidia-nvjitlink==$CUDA_NVCC_VERSION.*
nvidia-cuda-cccl==$CUDA_CCCL_VERSION.*
nvidia-cuda-profiler-api==$CUDA_CCCL_VERSION.*
EOF
  # torch only bounds nvjitlink from below; bring it to the compiler's version
  uvpip -c "$CONSTRAINTS" "nvidia-nvjitlink==$CUDA_NVCC_VERSION.*"
  cat "$CONSTRAINTS"
}

# the compiler and headers next to torch's CUDA 13 libraries: a complete CUDA_HOME inside the environment
s_cuda_toolkit() {
  uvpip -c "$CONSTRAINTS" nvidia-cuda-nvcc nvidia-cuda-crt nvidia-nvvm nvidia-cuda-cccl nvidia-cuda-profiler-api \
    nvidia-cudnn-cu13 nvidia-cudnn-frontend ninja wheel "cmake==4.3.1" pybind11 scikit-build-core setuptools packaging
  local cuda_home
  cuda_home=$(site)/nvidia/cu13
  ln -sfn lib "$cuda_home/lib64"
  local library unversioned
  for library in "$cuda_home"/lib/lib*.so.[0-9]*; do
    unversioned=${library%%.so.*}.so
    [ -e "$unversioned" ] || ln -s "${library##*/}" "$unversioned"
  done
  # CUDA 13 no longer ships the NVTX 3 headers or cuda_profiler_api.h, which Transformer Engine includes
  mkdir -p "$cuda_home/include"
  cp -r "$PREFIX/src/NVTX/c/include/nvtx3" "$cuda_home/include/"
  [ -f "$cuda_home/include/cuda_profiler_api.h" ] ||
    cp "$PREFIX/src/slime/docker/patch/gb10/cuda_profiler_api.h" "$cuda_home/include/"
  "$cuda_home/bin/nvcc" --version | tail -2
}

s_python_deps() {
  uvpip -c "$CONSTRAINTS" -r "$PREFIX/src/slime/requirements.txt" "numpy==$NUMPY_VERSION" "scipy==$SCIPY_VERSION" \
    "pyarrow==$PYARROW_VERSION" "flash-linear-attention==$FLA_VERSION" "tilelang==$TILELANG_VERSION" "kernels<0.15.0" \
    "antlr4-python3-runtime==4.9.3" "nvidia-modelopt[torch]>=0.37.0"
  # async checkpointing (Megatron imports it on use). Its wheels are manylinux_2_39 (Ubuntu 24.04, as the NGC PyTorch and
  # slime images), so the build takes that wheel even on an older build host; there it does not load
  uvpip --no-deps --python-platform x86_64-manylinux_2_39 "nvidia-resiliency-ext==$NVIDIA_RESILIENCY_EXT_VERSION"
  uvpip --no-deps --reinstall "$SRC_IN/$(basename "$SGL_ROUTER_WHEEL_URL")"
}

build_env() {  # the variables of a CUDA extension build
  local cuda_home
  cuda_home=$(site)/nvidia/cu13
  export CUDA_HOME=$cuda_home CUDA_PATH=$cuda_home PATH=$ENV/bin:$cuda_home/bin:$PATH
  export TORCH_CUDA_ARCH_LIST=${CUDA_ARCHS//;/ } MAX_JOBS=$JOBS CMAKE_BUILD_PARALLEL_LEVEL=$JOBS
  export NVCC_APPEND_FLAGS="--threads 4" CMAKE_POLICY_VERSION_MINIMUM=3.5
  export CPATH="$cuda_home/include:$cuda_home/include/cccl${CPATH:+:$CPATH}"
  export LIBRARY_PATH="$cuda_home/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  local cudnn
  cudnn=$(site)/nvidia/cudnn
  export CUDNN_PATH=$cudnn CUDNN_HOME=$cudnn
}

s_transformer_engine() {
  build_env
  uv pip uninstall --python "$PY" transformer-engine transformer-engine-cu12 transformer-engine-cu13 \
    transformer-engine-torch 2> /dev/null || true
  NVTE_CUDA_ARCHS=$NVTE_CUDA_ARCHS NVTE_FRAMEWORK=pytorch NVTE_WITH_NCCL_EP=0 \
    uvpip -c "$CONSTRAINTS" --no-build-isolation "transformer_engine[pytorch,core_cu13]==$TRANSFORMER_ENGINE_VERSION"
}

s_apex() {
  build_env
  # apex's setup.py reads these as its --cpp_ext / --cuda_ext build options
  APEX_CPP_EXT=1 APEX_CUDA_EXT=1 uvpip -c "$CONSTRAINTS" --no-build-isolation --no-deps "$PREFIX/src/apex"
}

s_megatron() {
  build_env
  uvpip -c "$CONSTRAINTS" --no-build-isolation -e "$PREFIX/src/Megatron-LM"
}

s_torch_memory_saver() {
  build_env
  TMS_CUDA_MAJOR=13 uvpip -c "$CONSTRAINTS" --no-build-isolation --no-deps --reinstall "$PREFIX/src/torch_memory_saver"
}

s_mamba() {
  build_env
  CAUSAL_CONV1D_FORCE_BUILD=TRUE uvpip -c "$CONSTRAINTS" --no-build-isolation --no-deps \
    "causal-conv1d==$CAUSAL_CONV1D_VERSION"
  MAMBA_FORCE_BUILD=TRUE uvpip -c "$CONSTRAINTS" --no-build-isolation --no-deps "$PREFIX/src/mamba"
}

s_slime() {
  uvpip --no-deps -e "$PREFIX/src/slime"
  uvpip --no-deps --reinstall "nvidia-nccl-cu13==$NCCL_VERSION"
}

s_activate() {
  cat > "$PREFIX/activate.sh" <<EOF
# source $PREFIX/activate.sh
export SLIME_PREFIX=$PREFIX
export CUDA_HOME=\$(ls -d $ENV/lib/python3*/site-packages/nvidia/cu13)
export PATH=$ENV/bin:\$CUDA_HOME/bin:$PREFIX/bin:\$PATH
export PYTHONPATH=$PREFIX/src/Megatron-LM\${PYTHONPATH:+:\$PYTHONPATH}
# NCCL and cuDNN from the env before any system copy (NGC images put cuDNN 9.10 on the library path; TE needs >= 9.18.1
# for packed sequences on sm120)
export LD_LIBRARY_PATH=\$(ls -d $ENV/lib/python3*/site-packages/nvidia/nccl/lib 2>/dev/null || true):\$(ls -d $ENV/lib/python3*/site-packages/nvidia/cudnn/lib 2>/dev/null || true):\$CUDA_HOME/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
# Transformer Engine looks for cuDNN under CUDNN_PATH / CUDNN_HOME first
export CUDNN_PATH=\$(ls -d $ENV/lib/python3*/site-packages/nvidia/cudnn 2>/dev/null || true)
export CUDNN_HOME=\$CUDNN_PATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
EOF
  cp "$HERE/versions.env" "$PREFIX/versions.env"
  cp "$SRC_IN/SHA256SUMS" "$PREFIX/SOURCES.SHA256SUMS"
  uv pip freeze --python "$PY" > "$PREFIX/LOCK.txt"
  echo "$(wc -l < "$PREFIX/LOCK.txt") packages in $PREFIX/LOCK.txt"
}

step verify s_verify
step uv s_uv
step python s_python
step sources s_sources
step venv s_venv
step rust s_rust
step sglang s_sglang
step cuda_toolkit s_cuda_toolkit
step python_deps s_python_deps
step transformer_engine s_transformer_engine
step apex s_apex
step megatron s_megatron
step torch_memory_saver s_torch_memory_saver
step mamba s_mamba
step slime s_slime
step activate s_activate
echo "BUILD_OK $PREFIX"
