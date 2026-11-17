#!/usr/bin/env bash
# The built environment as one tar, unpacked with `tar -C / -xf <tar>` to the same $PREFIX on another machine
# (the virtual environment's paths are absolute). Leaves out the download cache and the build stamps.
#   bash env/slime/pack.sh <out tar>
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd -P)
# shellcheck source=versions.env
source "$HERE/versions.env"
OUT=${1:?out tar}
tar -C / --exclude="${PREFIX#/}/.cache" --exclude="${PREFIX#/}/stamps" --exclude="${PREFIX#/}/rust" \
  --exclude='__pycache__' --exclude="${PREFIX#/}/src/sglang/rust/target" \
  -cf "$OUT.part" "${PREFIX#/}"
mv "$OUT.part" "$OUT"
sha256sum "$OUT" | tee "$OUT.sha256"
du -h "$OUT"
