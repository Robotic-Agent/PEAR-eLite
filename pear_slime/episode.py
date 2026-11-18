"""PEAR LIBERO episodes as the token sequence of inference, without a training framework.

pear_libero_eval.py builds a turn's prompt by text concatenation, not by re-rendering the whole conversation:
    prompt_0   = render([system, user_0], add_generation_prompt=True)
    prompt_k+1 = prompt_k + text_k + "<|im_end|>" + "\n" + tail(user_k+1)
where text_k is the generated text (the thinking, "</think>", the answer; no eos) and tail(user) is
render([system, user], True) with the system part cut off. Only the text_k + "<|im_end|>" segments carry loss.

This is the logic of scripts/pear_episode_sft_dataset.py (verl), shared here with the slime rollout.
A row is the output of scripts/prepare_effort_sft.py: messages = system, user_0, assistant_0, user_1, ...; images in
order, two per user turn (agentview, wrist), referenced by "<image>" placeholders in the user content.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

IM_END = "<|im_end|>"
_CAMERA_LABEL = re.compile(r"\n?(\[(?:agentview|wrist)\])")


def load_image(item: Any) -> Image.Image:
    if isinstance(item, dict):
        item = item.get("bytes", item)
    if isinstance(item, (bytes, bytearray)):
        return Image.open(io.BytesIO(item)).convert("RGB")
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    raise TypeError(f"unsupported image entry {type(item)}")


def user_blocks(content: str, images: list[Image.Image]) -> list[dict]:
    """The user content as pear_libero_eval.user_content builds it: the turn text, then per camera a "[name]" text
    block and its image. Accepts both "\\n[agentview]<image>" (older rows) and "[agentview]<image>"."""
    blocks: list[dict] = []
    it = iter(images)
    for piece in re.split(r"(<image>)", content):
        if piece == "<image>":
            blocks.append({"type": "image", "image": next(it)})
            continue
        for part in _CAMERA_LABEL.split(piece):
            if part:
                blocks.append({"type": "text", "text": part})
    rest = list(it)
    if rest:
        raise ValueError(f"{len(rest)} images left over after the user content's <image> placeholders")
    return blocks


@dataclass
class Segment:
    text: str
    images: list[Image.Image]
    loss: bool


def episode_segments(processor, messages: list[dict], images: list[Any], chat_template_kwargs: dict) -> list[Segment]:
    def render(conversation: list[dict]) -> str:
        kwargs = {"enable_thinking": True, **chat_template_kwargs}
        return processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, **kwargs)

    images = [load_image(x) for x in images]
    if messages[0]["role"] != "system":
        raise ValueError("an episode row starts with its system message")
    system = {"role": "system", "content": messages[0]["content"]}

    segments: list[Segment] = []
    offset = 0
    system_render = None
    for message in messages[1:]:
        if message["role"] == "user":
            n = message["content"].count("<image>")
            turn_images = images[offset : offset + n]
            offset += n
            full = render([system, {"role": "user", "content": user_blocks(message["content"], turn_images)}])
            if system_render is None:
                system_render = full[: full.index("<|im_start|>user")]
                segments.append(Segment(full, turn_images, False))
            else:
                if not full.startswith(system_render):
                    raise ValueError("the system part of a render changed between turns")
                segments.append(Segment("\n" + full[len(system_render) :], turn_images, False))
        elif message["role"] == "assistant":
            text = message["content"]
            if not text.endswith(IM_END):
                text += IM_END
            segments.append(Segment(text, [], True))
        else:
            raise ValueError(f"unexpected role {message['role']}")
    if offset != len(images):
        raise ValueError(f"used {offset} of {len(images)} images")
    return segments


def encode_episode(processor, segments: list[Segment]) -> tuple[list[int], list[int], dict[str, torch.Tensor]]:
    """Token ids, per-token loss mask and the vision inputs (pixel_values, image_grid_thw in image order)."""
    token_ids: list[int] = []
    loss_mask: list[int] = []
    vision: dict[str, list[torch.Tensor]] = {}
    for segment in segments:
        if segment.images:
            out = processor(text=[segment.text], images=segment.images, return_tensors="pt")
            ids = out["input_ids"][0].tolist()
            for key in ("pixel_values", "image_grid_thw"):
                vision.setdefault(key, []).append(out[key])
        else:
            ids = processor.tokenizer(segment.text, add_special_tokens=False)["input_ids"]
        token_ids += ids
        loss_mask += [int(segment.loss)] * len(ids)
    return token_ids, loss_mask, {k: torch.cat(v, dim=0) for k, v in vision.items()}
