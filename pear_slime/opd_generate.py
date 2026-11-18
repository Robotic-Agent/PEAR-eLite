"""slime generate function for PEAR ReOPD: a pear_slime.opd_data row (episode context up to a user turn, its images)
becomes the inference prompt text exactly as pear_libero_eval.py builds it (pear_slime.episode), with the images as
sample.multimodal_inputs; slime's own SGLang generate then runs the student on it.

The thinking effort comes from --apply-chat-template-kwargs '{"reasoning_effort": ...}' (xhigh, medium, low), so a
curriculum is a sequence of runs with decreasing effort. --rollout-max-response-len is the thinking budget: a turn that
reaches it without closing its thinking gets "\\n</think>\\n\\n" appended (no loss on those tokens) and then writes its
action within PEAR_OPD_SOLUTION_BUDGET tokens (default 512), so the teacher still scores an action that follows a
truncated thought (the budget-constrained rollout of Elastic Reasoning, Xu et al., ICLR 2026).
Launch with: --custom-generate-function-path pear_slime.opd_generate.generate --input-key messages
    --metadata-key metadata (and no --apply-chat-template: the prompt stays the message list until here)
"""

from __future__ import annotations

import logging
import os

from slime.rollout.sglang_rollout import generate as slime_generate
from slime.utils.http_utils import post
from slime.utils.processing_utils import encode_image_for_rollout_engine, load_processor
from slime.utils.types import Sample

from pear_slime.episode import episode_segments, load_image

logger = logging.getLogger(__name__)
PROCESSOR = None
PRINTED = False
THINK_END = "\n</think>\n\n"
SOLUTION_BUDGET = int(os.environ.get("PEAR_OPD_SOLUTION_BUDGET", "512"))
FORCED = {"n": 0, "total": 0}


def build_prompt(args, sample) -> None:
    global PROCESSOR, PRINTED
    if PROCESSOR is None:
        PROCESSOR = load_processor(args.hf_checkpoint, trust_remote_code=True)
    kwargs = dict(args.apply_chat_template_kwargs or {})
    images = [load_image(x) for x in sample.metadata.pop("images")]
    segments = episode_segments(PROCESSOR, list(sample.prompt), images, kwargs)
    # the context's earlier assistant turns are loss segments of an SFT row; the prompt must end with the user turn
    if segments[-1].loss:
        raise ValueError("an OPD prompt must end with a user turn")
    sample.prompt = "".join(s.text for s in segments)
    sample.multimodal_inputs = {"images": images}
    if not PRINTED:
        logger.info("pear_slime.opd_generate first prompt: %d chars, %d images, turn %s, effort %s",
                    len(sample.prompt), len(images), sample.metadata.get("turn"), kwargs.get("reasoning_effort"))
        PRINTED = True


async def close_thinking(args, sample, sampling_params) -> Sample:
    """Append the end of thinking to a truncated turn and let the student write its action after it."""
    tokenizer = PROCESSOR.tokenizer
    forced = tokenizer(THINK_END, add_special_tokens=False)["input_ids"]
    sample.append_response_tokens(args, tokens=forced, trainable=False, text=THINK_END, update_terminal_info=False)
    params = dict(sampling_params)
    params["max_new_tokens"] = SOLUTION_BUDGET
    payload = {
        "text": sample.prompt + sample.response,
        "image_data": [encode_image_for_rollout_engine(image) for image in sample.multimodal_inputs["images"]],
        "sampling_params": params,
        "return_logprob": True,
    }
    output = await post(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate", payload)
    items = output["meta_info"].get("output_token_logprobs", [])
    sample.append_response_tokens(
        args,
        tokens=[item[1] for item in items],
        log_probs=[item[0] for item in items],
        trainable=True,
        meta_info=output["meta_info"],
        text=output["text"],
    )
    return sample


async def generate(args, sample, sampling_params):
    # an aborted sample comes back with its prompt already built
    if not isinstance(sample.prompt, str):
        build_prompt(args, sample)
    sample = await slime_generate(args, sample, dict(sampling_params))
    FORCED["total"] += 1
    if sample.status == Sample.Status.TRUNCATED and "</think>" not in (sample.response or "") and SOLUTION_BUDGET > 0:
        sample = await close_thinking(args, sample, sampling_params)
        FORCED["n"] += 1
        if FORCED["n"] in (1, 10, 100) or FORCED["n"] % 500 == 0:
            logger.info("pear_slime.opd_generate: thinking closed by force in %d of %d turns", FORCED["n"],
                        FORCED["total"])
    return sample
