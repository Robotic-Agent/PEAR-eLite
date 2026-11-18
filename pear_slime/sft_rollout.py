"""slime SFT rollout for PEAR LIBERO episodes (multi-turn, two camera images per user turn).

slime's slime.rollout.sft_rollout tokenizes text only. This one builds the inference token sequence with
pear_slime.episode, puts loss on the generated turns, and hands the vision inputs to Qwen3.5-VL through
Sample.multimodal_train_inputs.

Dataset rows (pear_slime/convert_sft_data.py): messages (list of {role, content}), metadata = {"images": [bytes, ...],
"id": ...}. Launch with:
    --rollout-function-path pear_slime.sft_rollout.generate_rollout --input-key messages --metadata-key metadata
    --apply-chat-template-kwargs '{"reasoning_effort": "xhigh"}'
"""

from __future__ import annotations

import logging
import os
import random

from slime.utils.processing_utils import load_processor

from pear_slime.augment import augment_episode
from pear_slime.episode import encode_episode, episode_segments, load_image

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)
# conservative image augmentation (pear_slime/augment.py): the probability an episode is augmented, 0 = off
AUG_P = float(os.environ.get("PEAR_SFT_AUG", "0"))
AUG_RNG = random.Random(int(os.environ.get("PEAR_SFT_AUG_SEED", "0")))

PROCESSOR = None
PRINTED = False


def build_sample(sample, processor, chat_template_kwargs: dict):
    images = sample.metadata.pop("images")
    if AUG_P > 0:
        images = augment_episode([load_image(x) for x in images], AUG_P, AUG_RNG)
    segments = episode_segments(processor, list(sample.prompt), list(images), chat_template_kwargs)
    token_ids, loss_mask, vision = encode_episode(processor, segments)
    first = loss_mask.index(1)
    sample.tokens = token_ids
    sample.response_length = len(token_ids) - first
    sample.loss_mask = loss_mask[first:]
    sample.reward = 0
    sample.multimodal_train_inputs = vision or None
    return sample


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    assert not evaluation, "PEAR SFT rollout has no evaluation mode"
    global PROCESSOR, PRINTED
    if PROCESSOR is None:
        PROCESSOR = load_processor(args.hf_checkpoint, trust_remote_code=True)
    kwargs = dict(args.apply_chat_template_kwargs or {})

    samples = data_buffer.get_samples(args.rollout_batch_size)
    out = []
    for group in samples:
        (sample,) = group
        # the data source hands out deep copies, so the image bytes are dropped here and do not travel to training
        built = build_sample(sample, PROCESSOR, kwargs)
        if not PRINTED:
            grid = built.multimodal_train_inputs.get("image_grid_thw") if built.multimodal_train_inputs else None
            logger.info(
                "pear_slime.sft_rollout first sample: %d tokens, %d with loss, response_length %d, images %s",
                len(built.tokens), sum(built.loss_mask), built.response_length,
                None if grid is None else tuple(grid.shape),
            )
            PRINTED = True
        out.append(group)
    return out
