#!/usr/bin/env bash
# Everything the build needs from GitHub, into one directory: run on a machine with GitHub access, then copy the
# directory to the build machine (which only needs PyPI). SHA256SUMS records what was fetched.
#   bash env/slime/fetch_sources.sh <out dir>
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd -P)
# shellcheck source=versions.env
source "$HERE/versions.env"
OUT=${1:?out dir}
mkdir -p "$OUT"
cd "$OUT"

get() {  # url file
  [ -s "$2" ] && return 0
  curl -fL --retry 5 --retry-delay 3 -o "$2.part" "$1" && mv "$2.part" "$2"
  echo "fetched $2"
}
gh_tar() {  # owner/repo ref name
  get "https://codeload.github.com/$1/tar.gz/$2" "$3-$2.tar.gz"
}

get "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz" uv-x86_64-unknown-linux-gnu.tar.gz
echo "$UV_SHA256_x86_64  uv-x86_64-unknown-linux-gnu.tar.gz" | shasum -a 256 -c -
gh_tar THUDM/slime "$SLIME_COMMIT" slime
gh_tar sgl-project/sglang "v$SGLANG_VERSION" sglang
gh_tar NVIDIA/Megatron-LM "$MEGATRON_COMMIT" Megatron-LM
gh_tar NVIDIA/apex "$APEX_COMMIT" apex
gh_tar zhuzilin/torch_memory_saver "$TMS_COMMIT" torch_memory_saver
gh_tar state-spaces/mamba "$MAMBA_COMMIT" mamba
gh_tar NVIDIA/NVTX "$NVTX_COMMIT" NVTX
get "$SGL_ROUTER_WHEEL_URL" "$(basename "$SGL_ROUTER_WHEEL_URL")"

shasum -a 256 ./*.tar.gz ./*.whl | sed 's#  \./#  #' > SHA256SUMS
cat SHA256SUMS
