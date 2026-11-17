"""Check PearEpisodeSFTDataset against what vLLM saw at inference.

For every row: the token count before each assistant segment must equal that turn's prompt_tokens in
<run>/turns/<suite>_<task>_<trial>.jsonl, and every assistant segment's token count is compared with the turn's
generated count ("gen", which excludes the eos that the segment adds).
usage: python check_episode_sft_dataset.py <parquet> <model dir> <effort> <run dir> [<run dir> ...]
"""
import importlib.util
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf
from transformers import AutoProcessor

parquet, model, effort, *runs = sys.argv[1:]
spec = importlib.util.spec_from_file_location("ds", str(Path(__file__).with_name("pear_episode_sft_dataset.py")))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
config = OmegaConf.create({
    "max_length": 10**7, "pad_mode": "no_padding", "truncation": "error",
    "apply_chat_template_kwargs": {"reasoning_effort": effort},
})
ds = mod.PearEpisodeSFTDataset(parquet_files=parquet, tokenizer=processor.tokenizer, config=config, processor=processor)
print("rows", len(ds))

bad = 0
for i in range(len(ds)):
    row = ds.dataframe.iloc[i].to_dict()
    name = f"{row['suite']}_{row['task_id']}_{row['trial_id']}"
    turns_file = next((Path(r) / "turns" / f"{name}.jsonl" for r in runs if (Path(r) / "turns" / f"{name}.jsonl").exists()), None)
    turns = [json.loads(x) for x in turns_file.read_text().splitlines()]
    segments = ds._build_messages(row)
    pos, k, mism = 0, 0, []
    gen_diff = []
    for j, seg in enumerate(segments):
        ids, loss, _, _ = ds._process_single_message(j, seg, segments)
        if seg["loss"]:
            if pos != turns[k]["prompt_tokens"]:
                mism.append((k, pos, turns[k]["prompt_tokens"]))
            gen_diff.append(len(ids) - 1 - turns[k]["gen"])
            k += 1
        pos += len(ids)
    item = ds[i]
    n_loss = int(item["loss_mask"].sum())
    status = "OK" if not mism and k == len(turns) else "MISMATCH"
    bad += status != "OK"
    print(f"{i:3d} {name:20s} turns={k}/{len(turns)} tokens={len(item['input_ids'])} loss_tokens={n_loss} "
          f"gen_diff(min/max)={min(gen_diff)}/{max(gen_diff)} {status} {mism[:3]}")
    if i == 0:
        first = processor.tokenizer.decode(ds._process_single_message(0, segments[0], segments)[0])
        example = Path(turns_file).parents[1] / "prompt_example.txt"
        print("first segment decodes to the saved prompt_example:", example.exists() and first.replace(
            "<|image_pad|>", "") == example.read_text().replace("<|image_pad|>", ""))
print("BAD ROWS", bad)
