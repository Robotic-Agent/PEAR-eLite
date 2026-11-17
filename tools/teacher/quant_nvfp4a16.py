"""NVFP4 weight-only (W4A16) quantization of the PEAR-27B teacher, data-free, on CPU.

Quantizes the language model's self_attn and mlp Linear layers; keeps lm_head, the vision tower and the
linear-attention (Gated DeltaNet) projections in BF16.
usage: $WORK_DIR/envs/quant/bin/python tools/teacher/quant_nvfp4a16.py <src model dir> <out dir>
"""
import json
import os
import shutil
import sys
import time

import torch
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoConfig, AutoModelForImageTextToText

src, out = sys.argv[1], sys.argv[2]
t0 = time.time()
cfg = AutoConfig.from_pretrained(src)
print("architectures", cfg.architectures, flush=True)
model = AutoModelForImageTextToText.from_pretrained(src, dtype=torch.bfloat16, device_map="cpu")
print(f"loaded in {time.time() - t0:.0f} s; class {type(model).__name__}", flush=True)

linear_names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
targets = [n for n in linear_names if (".self_attn." in n or ".mlp." in n) and "visual" not in n]
print(f"{len(linear_names)} Linear modules, {len(targets)} to quantize; e.g. {targets[:2]}", flush=True)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4A16",
    ignore=["lm_head", "re:.*visual.*", "re:.*linear_attn.*"],
)
oneshot(model=model, recipe=recipe)
print(f"quantized in {time.time() - t0:.0f} s", flush=True)

model.save_pretrained(out, save_compressed=True)
for f in os.listdir(src):
    if f.endswith((".safetensors", ".bin")) or f in ("model.safetensors.index.json", "config.json"):
        continue
    p = os.path.join(src, f)
    if os.path.isfile(p) and not os.path.exists(os.path.join(out, f)):
        shutil.copy(os.path.realpath(p), os.path.join(out, f))
qc = json.load(open(os.path.join(out, "config.json"))).get("quantization_config", {})
print("quantization_config keys", sorted(qc), "format", qc.get("format"), "ignore", qc.get("ignore", [])[:4], flush=True)
print(f"saved to {out} in {time.time() - t0:.0f} s", flush=True)
print("QUANT_OK", flush=True)
