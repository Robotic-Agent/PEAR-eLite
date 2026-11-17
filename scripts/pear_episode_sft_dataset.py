"""verl SFT dataset (data.custom_cls) for PEAR LIBERO episodes that reproduces the token sequence of inference.

pear_libero_eval.py builds a turn's prompt by text concatenation, not by re-rendering the whole conversation:
    prompt_0   = render([system, user_0], add_generation_prompt=True)
    prompt_k+1 = prompt_k + text_k + "<|im_end|>" + "\n" + tail(user_k+1)
where text_k is vLLM's generated text (the thinking, "</think>", the answer; no eos) and tail(user) is
render([system, user], True) with the system part cut off. verl's MultiTurnSFTDataset instead renders every message
alone with the chat template, which this template rejects ("No user query found") and which would put an empty
<think></think> before text that already holds the thinking.

Each row of scripts/prepare_effort_sft.py (messages = system, user_0, assistant_0, user_1, ...; images = 2 per user) is
turned into segments; only the text_k + "<|im_end|>" segments carry loss. The rest of MultiTurnSFTDataset.__getitem__
(position ids for Qwen-VL, padding) is reused unchanged by overriding its three per-message hooks.
"""

from __future__ import annotations

import io
import re
from typing import Any, Optional

import torch
from PIL import Image

import verl.utils.dataset.multiturn_sft_dataset as _multiturn
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

IM_END = "<|im_end|>"

_printed = False


def _print_assembled_segments(tokenizer, message_list, input_ids, loss_mask, attn_mask, tools):
    """Replaces verl's print_assembled_message, which re-renders message_list with the chat template (segments here)."""
    global _printed
    if _printed:
        return
    _printed = True
    n_loss = int(loss_mask.sum())
    print(f"[PearEpisodeSFTDataset] first row: {len(input_ids)} tokens, {n_loss} with loss, {len(message_list)} segments")


_multiturn.print_assembled_message = _print_assembled_segments
_CAMERA_LABEL = re.compile(r"\n?(\[(?:agentview|wrist)\])")


def _image(item: Any) -> Image.Image:
    if isinstance(item, dict):
        item = item.get("bytes", item)
    if isinstance(item, (bytes, bytearray)):
        return Image.open(io.BytesIO(item)).convert("RGB")
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    raise TypeError(f"unsupported image entry {type(item)}")


def _user_blocks(content: str, images: list[Image.Image]) -> list[dict]:
    """The user content as pear_libero_eval.user_content builds it: the turn text, then per camera a "[name]" text
    block and its image. Accepts both "\\n[agentview]<image>" (older prepare_effort_sft rows) and "[agentview]<image>"."""
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


class PearEpisodeSFTDataset(MultiTurnSFTDataset):
    def _render(self, messages: list[dict]) -> str:
        kwargs = {"enable_thinking": True, **self.apply_chat_template_kwargs}
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)

    def _build_messages(self, example: dict) -> list[dict]:
        """Returns the episode's segments: {"role": "segment", "text", "images", "loss"}."""
        from verl.utils.dataset.multiturn_sft_dataset import convert_nested_value_to_list_recursive

        messages = convert_nested_value_to_list_recursive(example[self.messages_key])
        images = [_image(x) for x in convert_nested_value_to_list_recursive(example[self.image_key])]
        if messages[0]["role"] != "system":
            raise ValueError("an episode row starts with its system message")
        system = {"role": "system", "content": messages[0]["content"]}

        segments: list[dict] = []
        offset = 0
        system_render = None
        for message in messages[1:]:
            if message["role"] == "user":
                n = message["content"].count("<image>")
                user_images = images[offset : offset + n]
                offset += n
                full = self._render([system, {"role": "user", "content": _user_blocks(message["content"], user_images)}])
                if system_render is None:
                    system_render = full[: full.index("<|im_start|>user")]
                    segments.append({"role": "segment", "text": full, "images": user_images, "loss": False})
                else:
                    if not full.startswith(system_render):
                        raise ValueError("the system part of a render changed between turns")
                    segments.append(
                        {"role": "segment", "text": "\n" + full[len(system_render) :], "images": user_images,
                         "loss": False}
                    )
            elif message["role"] == "assistant":
                text = message["content"]
                if not text.endswith(IM_END):
                    text += IM_END
                segments.append({"role": "segment", "text": text, "images": [], "loss": True})
            else:
                raise ValueError(f"unexpected role {message['role']}")
        if offset != len(images):
            raise ValueError(f"used {offset} of {len(images)} images")
        return segments

    def _process_single_message(
        self,
        index: int,
        message: dict[str, Any],
        full_message: list,
        tools: Optional[list[dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = None,
    ):
        if message["images"]:
            inputs = self.processor(text=[message["text"]], images=message["images"], return_tensors="pt")
        else:
            inputs = self.processor.tokenizer([message["text"]], add_special_tokens=False, return_tensors="pt")
        inputs = dict(inputs)
        input_ids = inputs.pop("input_ids")[0]
        attention_mask = inputs.pop("attention_mask")[0]
        loss_mask = torch.ones_like(attention_mask) if message["loss"] else torch.zeros_like(attention_mask)
        return input_ids, loss_mask, attention_mask, inputs

    def sanity_check(self, input_ids, messages, tools, enable_thinking):
        return None  # the reference is the inference prompt, checked by scripts/check_episode_sft_dataset.py
