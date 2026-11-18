"""Token count of each SFT row (chat template text + image tokens from the processor) and a <= limit filter,
without verl. usage: count_filter_tokens.py <in parquet> <model dir> <limit> <out parquet>"""
import io, json, sys
import pyarrow as pa, pyarrow.parquet as pq
from PIL import Image
from transformers import AutoProcessor

src, model, limit, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
proc = AutoProcessor.from_pretrained(model, trust_remote_code=True)
tok = proc.tokenizer
pad = proc.image_token if hasattr(proc, "image_token") else "<|image_pad|>"
f = pq.ParquetFile(src)
cache, counts = {}, []
for g in range(f.num_row_groups):
    for r in f.read_row_group(g, columns=["messages", "images"]).to_pylist():
        msgs = [{"role": m["role"], "content": m["content"].replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")} for m in r["messages"]]
        text = tok.apply_chat_template(msgs, tokenize=False, reasoning_effort="xhigh", enable_thinking=True)
        n_text = len(tok(text, add_special_tokens=False)["input_ids"])
        n_img = 0
        for im in r["images"]:
            size = Image.open(io.BytesIO(im["bytes"])).size
            if size not in cache:
                grid = proc.image_processor(images=[Image.open(io.BytesIO(im["bytes"])).convert("RGB")], return_tensors="pt")["image_grid_thw"][0]
                merge = getattr(proc.image_processor, "merge_size", 2)
                cache[size] = int(grid.prod()) // (merge * merge)
            n_img += cache[size] - 1  # the placeholder pad already counted once
        counts.append(n_text + n_img)
print("image tokens by size", cache)
keep = [i for i, c in enumerate(counts) if c <= limit]
print(f"rows {len(counts)} keep {len(keep)} dropped {len(counts) - len(keep)} max {max(counts)} mean {sum(counts) / len(counts):.0f}")
keep_set, start = set(keep), 0
with pq.ParquetWriter(out, f.schema_arrow) as w:
    for g in range(f.num_row_groups):
        part = f.read_row_group(g)
        idx = [i - start for i in range(start, start + part.num_rows) if i in keep_set]
        if idx:
            w.write_table(part.take(pa.array(idx)))
        start += part.num_rows
json.dump({"limit": limit, "counts": counts, "kept": len(keep)}, open(out + ".filter.json", "w"))
