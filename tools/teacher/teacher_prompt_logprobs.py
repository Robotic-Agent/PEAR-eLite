"""Top-k prompt log-probs of a teacher on the texts of an OPD rollout dump (text only, no images).
usage: python teacher_prompt_logprobs.py <model dir> <tp> <rollout dump jsonl> <out json> [max tokens per text]
"""
import json
import sys
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

K = 32


def main():
    model, tp, dump, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 6000

    tok = AutoTokenizer.from_pretrained(model)
    rows = [json.loads(line) for line in open(dump)]
    prompts = []
    for r in rows:
        ids = tok(r["input"] + r["output"], add_special_tokens=False)["input_ids"][:max_tokens]
        prompts.append({"prompt_token_ids": ids})
    print(f"{len(prompts)} texts, tokens {[len(p['prompt_token_ids']) for p in prompts]}", flush=True)

    t0 = time.time()
    llm = LLM(model=model, tensor_parallel_size=tp, max_model_len=max_tokens + 16, gpu_memory_utilization=0.65,
              max_logprobs=K, enable_prefix_caching=False, limit_mm_per_prompt={"image": 0},
              max_num_batched_tokens=1024, enable_chunked_prefill=True, max_num_seqs=2)
    print(f"loaded in {time.time() - t0:.0f} s", flush=True)
    res = llm.generate(prompts, SamplingParams(max_tokens=1, prompt_logprobs=K))
    data = []
    for p, r in zip(prompts, res):
        seq = []
        for pos, lp in enumerate(r.prompt_logprobs):
            if lp is None:
                continue
            seq.append({str(t): v.logprob for t, v in lp.items()} | {"_tok": p["prompt_token_ids"][pos]})
        data.append(seq)
    json.dump(data, open(out, "w"))
    print(f"PROMPT_LOGPROBS_OK {out} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
