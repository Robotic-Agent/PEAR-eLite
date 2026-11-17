"""Merge a PEFT LoRA adapter (exported by scripts/export_lora_adapter.py) into a Hugging Face model's safetensors
shards: W <- W + (lora_alpha / r) * B @ A, computed in fp32 and stored in the base weight's dtype. Every adapter pair
must map to a base weight, otherwise it stops. Non-weight files of the base directory are copied (symlinks resolved).
usage: python merge_lora_shards.py <base model dir> <adapter dir> <output dir>
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

base, adapter, out = (Path(p) for p in sys.argv[1:4])
out.mkdir(parents=True, exist_ok=False)
cfg = json.loads((adapter / "adapter_config.json").read_text())
scale = cfg["lora_alpha"] / cfg["r"]

pairs: dict[str, dict[str, torch.Tensor]] = {}
with safe_open(str(adapter / "adapter_model.safetensors"), "pt") as f:
    for key in f.keys():
        name = key.removeprefix("base_model.model.")
        for part in ("lora_A", "lora_B"):
            tag = f".{part}.weight"
            if name.endswith(tag):
                target = name[: -len(tag)] + ".weight"
                pairs.setdefault(target, {})[part] = f.get_tensor(key)
assert all(len(v) == 2 for v in pairs.values()), "an adapter pair is incomplete"

index = json.loads((base / "model.safetensors.index.json").read_text())
weight_map = index["weight_map"]
missing = [k for k in pairs if k not in weight_map]
assert not missing, f"{len(missing)} adapter targets have no base weight, e.g. {missing[:3]}"

merged = 0
max_rel = 0.0
for shard in sorted(set(weight_map.values())):
    tensors = {}
    with safe_open(str(base / shard), "pt") as f:
        metadata = f.metadata()
        for key in f.keys():
            w = f.get_tensor(key)
            if key in pairs:
                a, b = pairs[key]["lora_A"].float(), pairs[key]["lora_B"].float()
                delta = (b @ a) * scale
                assert delta.shape == w.shape, f"{key}: delta {tuple(delta.shape)} vs weight {tuple(w.shape)}"
                new = (w.float() + delta).to(w.dtype)
                max_rel = max(max_rel, (delta.norm() / w.float().norm()).item())
                w = new
                merged += 1
            tensors[key] = w.contiguous()
    save_file(tensors, str(out / shard), metadata=metadata or {"format": "pt"})
    print(f"wrote {shard}", flush=True)

for item in base.iterdir():
    if item.name.endswith(".safetensors") or item.is_dir():
        continue
    shutil.copy(item.resolve(), out / item.name)
(out / "lora_merge.json").write_text(json.dumps({
    "base": str(base), "adapter": str(adapter), "r": cfg["r"], "lora_alpha": cfg["lora_alpha"],
    "merged_weights": merged, "adapter_pairs": len(pairs), "max_relative_delta_norm": max_rel,
}, indent=2) + "\n")
print(f"merged {merged}/{len(pairs)} weights, max |delta|/|W| = {max_rel:.4g}")
assert merged == len(pairs)
