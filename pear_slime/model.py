"""slime's Qwen3.5-VL model provider with activation checkpointing in the replicated Transformers ViT.

An episode carries up to ~30 images whose ViT activations are kept for the backward pass on every rank; recomputing
them per block keeps the vision tower trainable within the 32 GB of an RTX 5090.
"""

from __future__ import annotations

from slime_plugins.models.qwen3_5_vl import get_qwen3_5_vl_model_provider as _slime_provider


def get_qwen3_5_vl_model_provider(args, config, vp_stage):
    build = _slime_provider(args, config, vp_stage)

    def model_provider(pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None):
        model = build(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        if model.model.visual is not None:
            model.model.visual.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return model

    return model_provider
