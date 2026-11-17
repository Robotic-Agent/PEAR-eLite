"""Drop the SFT rows whose full token count (from check_episode_sft_dataset.py output) exceeds a limit.
usage: python filter_sft_rows.py <in parquet> <check.txt> <max tokens> <out parquet>
"""
import json
import re
import sys

import pyarrow as pa
import pyarrow.parquet as pq

src, check, limit, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
tokens = {}
for line in open(check):
    m = re.match(r"^\s*(\d+) +(\S+) +turns=\S+ tokens=(\d+)", line)
    if m:
        tokens[int(m.group(1))] = (m.group(2), int(m.group(3)))
f = pq.ParquetFile(src)
assert len(tokens) == f.metadata.num_rows, (len(tokens), f.metadata.num_rows)
keep = sorted(i for i, (_, t) in tokens.items() if t <= limit)
dropped = [tokens[i] for i in sorted(tokens) if tokens[i][1] > limit]
# row group by row group: one table of all rows overflows the 2 GiB offsets of the image column
keep_set, start = set(keep), 0
with pq.ParquetWriter(out, f.schema_arrow) as writer:
    for g in range(f.num_row_groups):
        part = f.read_row_group(g)
        idx = [i - start for i in range(start, start + part.num_rows) if i in keep_set]
        start += part.num_rows
        if idx:
            writer.write_table(part.take(pa.array(idx)), row_group_size=16)
json.dump({"source": src, "limit": limit, "rows_in": len(tokens), "rows_out": len(keep), "dropped": dropped},
          open(out + ".filter.json", "w"), indent=2)
print(f"kept {len(keep)}/{len(tokens)}, dropped {dropped}")
