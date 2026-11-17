#!/usr/bin/env python3
"""Export a PEFT adapter from a verl FSDP LoRA-only checkpoint without writing the base model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--hf-model-config-path", type=Path)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_lora_metadata(checkpoint_dir: Path) -> dict:
    meta_path = checkpoint_dir / "lora_train_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"LoRA metadata is missing: {meta_path}")
    metadata = json.loads(meta_path.read_text())
    if int(metadata.get("r", 0)) <= 0:
        raise ValueError(f"LoRA rank must be positive in {meta_path}")
    if int(metadata.get("lora_alpha", 0)) <= 0:
        raise ValueError(f"LoRA alpha must be positive in {meta_path}")
    return metadata


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    output_root = args.output_root.resolve()
    hs_config_dir = (args.hf_model_config_path or checkpoint_dir / "huggingface").resolve()

    for path in (checkpoint_dir / "fsdp_config.json", hs_config_dir / "config.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = load_lora_metadata(checkpoint_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=str(checkpoint_dir),
        target_dir=str(output_root),
        hf_model_config_path=str(hs_config_dir),
        trust_remote_code=args.trust_remote_code,
    )
    merger = FSDPModelMerger(config)

    world_size = merger._get_world_size()
    rank_zero_state_dict = merger._load_rank_zero_state_dict(world_size)
    mesh, mesh_dim_names = merger._extract_device_mesh_info(rank_zero_state_dict, world_size)
    total_shards, mesh_shape = merger._calculate_shard_configuration(mesh, mesh_dim_names)
    state_dict = merger._load_and_merge_state_dicts(world_size, total_shards, mesh_shape, mesh_dim_names)

    adapter_keys = {key for key in state_dict if "lora_" in key or ".adapter_" in key}
    foreign_keys = set(state_dict).difference(adapter_keys)
    if foreign_keys:
        sample = ", ".join(sorted(foreign_keys)[:5])
        raise RuntimeError(
            f"expected a LoRA-only checkpoint, found {len(foreign_keys)} base-model keys: {sample}"
        )

    adapter_path = merger.save_lora_adapter(state_dict)
    if adapter_path is None:
        raise RuntimeError("the checkpoint contains no LoRA parameters")

    adapter_path = Path(adapter_path)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        path = adapter_path / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"adapter export did not produce {path}")

    print(
        json.dumps(
            {
                "adapter_path": str(adapter_path),
                "adapter_tensors": len(adapter_keys),
                "r": int(metadata["r"]),
                "lora_alpha": int(metadata["lora_alpha"]),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
