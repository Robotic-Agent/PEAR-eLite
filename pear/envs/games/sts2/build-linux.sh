#!/usr/bin/env bash
# Build the pinned emulator with the existing sts2-rl ELF link workaround, without its training image.
set -euo pipefail

engine=$(cd "${1:?usage: build-linux.sh <emulator checkout> <sts2-rl checkout>}" && pwd)
sts2_rl=$(cd "${2:?provide the sts2-rl checkout}" && pwd)
project="$engine/src/Sts2Emulator"
patch="$sts2_rl/patches/sts2emulator-linux-cross.patch"

# Apply to the external engine checkout only; accept the exact patch already being in place.
if ! git -C "$engine" apply --reverse --check "$patch" 2>/dev/null; then
  git -C "$engine" apply "$patch"
fi
dotnet publish "$project/Sts2Emulator.csproj" -c Release -r linux-x64 --self-contained -o "$engine/out"

# sts2-rl's original relinker assumes Docker's /root. Find the exact ILC package resolved by this build instead.
PEAR_STS2_ILC_ROOT=$(python3 - "$project/obj/project.assets.json" <<'PY'
import json
import sys
from pathlib import Path

assets = json.loads(Path(sys.argv[1]).read_text())
prefix = "runtime.linux-x64.microsoft.dotnet.ilcompiler/"
package = next(x["path"] for name, x in assets["libraries"].items() if name.lower().startswith(prefix))
print(next(p for folder in assets["packageFolders"] if (p := Path(folder) / package).is_dir()))
PY
)
export PEAR_STS2_ILC_ROOT
scratch=$(mktemp -d)
trap 'rm -f -- "$scratch/relink-linux.sh" "$scratch/elf-globalize.py"; rmdir -- "$scratch"' EXIT
sed 's|^ILC_ROOT=.*$|ILC_ROOT="${PEAR_STS2_ILC_ROOT}/"|' "$sts2_rl/scripts/relink-linux.sh" > "$scratch/relink-linux.sh"
cp "$sts2_rl/scripts/elf-globalize.py" "$scratch/elf-globalize.py"
bash "$scratch/relink-linux.sh" "$project" "$engine/out"
