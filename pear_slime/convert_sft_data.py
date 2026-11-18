"""The SFT parquet of scripts/prepare_effort_sft.py (verl: messages, images) as a slime prompt dataset:
messages unchanged, metadata = {"images": [encoded image bytes, in order], "row": index}.
Images are large_binary: the column exceeds 2 GiB, and slime reads the file with iter_batches, which cannot
return a nested binary column split into chunks.
usage: python -m pear_slime.convert_sft_data <in.parquet> <out.parquet>
"""

from __future__ import annotations

import sys

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema(
    [
        ("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("metadata", pa.struct([("images", pa.list_(pa.large_binary())), ("row", pa.int64())])),
    ]
)


def image_bytes(item) -> bytes:
    if isinstance(item, dict):
        item = item["bytes"]
    if not isinstance(item, (bytes, bytearray)):
        raise TypeError(f"expected encoded image bytes, got {type(item)}")
    return bytes(item)


def main(src: str, dst: str) -> None:
    source = pq.ParquetFile(src)
    writer = pq.ParquetWriter(dst + ".part", SCHEMA)
    row = n_images = 0
    for batch in source.iter_batches(batch_size=8, columns=["messages", "images"]):
        rows = []
        for record in batch.to_pylist():
            messages = [{"role": m["role"], "content": m["content"]} for m in record["messages"]]
            images = [image_bytes(x) for x in record["images"]]
            placeholders = sum(m["content"].count("<image>") for m in messages if m["role"] == "user")
            if placeholders != len(images):
                raise ValueError(f"row {row}: {placeholders} <image> placeholders, {len(images)} images")
            rows.append({"messages": messages, "metadata": {"images": images, "row": row}})
            row += 1
            n_images += len(images)
        writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
    writer.close()
    import os

    os.replace(dst + ".part", dst)
    print(f"{row} rows, {n_images} images -> {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
