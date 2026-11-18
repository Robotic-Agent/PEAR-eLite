"""Turn prompts for replayed on-policy distillation (ReOPD) from recorded PEAR episodes.

Each output row is one episode cut just before an assistant turn: messages = system, user_0, assistant_0, ...,
user_t (the recorded 27B turns as the context), metadata.images = the images of user_0..user_t. The student generates
turn t itself and the teacher scores it (pear_slime.opd_generate, pear_slime.opd_teacher); no simulator is needed.

Input: a pear_slime/convert_sft_data.py parquet (messages, metadata.images).
usage: python -m pear_slime.opd_data <in.parquet> <out.parquet> [--turns-per-episode 4] [--seed 0]
"""

from __future__ import annotations

import argparse
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema(
    [
        ("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        (
            "metadata",
            pa.struct([("images", pa.list_(pa.large_binary())), ("row", pa.int64()), ("turn", pa.int64())]),
        ),
    ]
)


def cut(messages: list[dict], images: list[bytes], turn: int) -> tuple[list[dict], list[bytes]]:
    """The episode up to and including the user message before assistant turn `turn` (0-based)."""
    out, seen, n_images = [], -1, 0
    for m in messages:
        if m["role"] == "assistant":
            seen += 1
            if seen == turn:
                break
        out.append(m)
        if m["role"] == "user":
            n_images += m["content"].count("<image>")
    if out[-1]["role"] != "user":
        raise ValueError("a cut ends with a user message")
    return out, images[:n_images]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--turns-per-episode", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = random.Random(a.seed)
    writer = pq.ParquetWriter(a.dst + ".part", SCHEMA)
    n_rows = n_episodes = 0
    for batch in pq.ParquetFile(a.src).iter_batches(batch_size=4, columns=["messages", "metadata"]):
        rows = []
        for record in batch.to_pylist():
            messages, images = record["messages"], record["metadata"]["images"]
            n_turns = sum(m["role"] == "assistant" for m in messages)
            turns = sorted(rng.sample(range(n_turns), min(a.turns_per_episode, n_turns)))
            for t in turns:
                msgs, imgs = cut(messages, images, t)
                rows.append({"messages": msgs, "metadata": {"images": imgs, "row": record["metadata"]["row"], "turn": t}})
            n_episodes += 1
        n_rows += len(rows)
        writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
    writer.close()
    os.replace(a.dst + ".part", a.dst)
    print(f"{n_episodes} episodes -> {n_rows} turn prompts -> {a.dst}")


if __name__ == "__main__":
    main()
