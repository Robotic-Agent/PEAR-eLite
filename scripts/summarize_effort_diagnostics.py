#!/usr/bin/env python3
"""Summarize PEAR effort/KV evaluation runs and perform paired comparisons."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def episode_key(record: dict) -> tuple[str, int, int]:
    env = record["env"]
    return env["suite"], int(env["task_id"]), int(env["trial_id"])


def load_runs(paths: list[Path]) -> dict[tuple[str, int, int], dict]:
    final = {}
    for path in paths:
        records = path / "episodes.jsonl" if path.is_dir() else path
        for line in records.read_text().splitlines():
            record = json.loads(line)
            key = episode_key(record)
            if key not in final or record.get("end_reason") != "error":
                final[key] = record
    return final


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return center - radius, center + radius


def exact_mcnemar(a_success_b_failure: int, a_failure_b_success: int) -> float:
    """Two-sided exact McNemar p-value, equivalent to a binomial sign test on discordant pairs."""
    discordant = a_success_b_failure + a_failure_b_success
    if discordant == 0:
        return 1.0
    smaller = min(a_success_b_failure, a_failure_b_success)
    tail = sum(math.comb(discordant, index) for index in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def summarize(records: dict[tuple[str, int, int], dict]) -> dict:
    rows = list(records.values())
    successes = sum(bool(row.get("success")) for row in rows)
    interval = wilson(successes, len(rows))
    chunks = sum(int(row.get("num_chunks", 0)) for row in rows)
    generated = sum(int(row.get("generated_tokens", 0)) for row in rows)
    return {
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows) if rows else None,
        "wilson_95": list(interval),
        "generated_tokens_per_round": generated / chunks if chunks else None,
        "mean_rounds": chunks / len(rows) if rows else None,
        "mean_rounds_success": (
            sum(int(row.get("num_chunks", 0)) for row in rows if row.get("success")) / successes
            if successes else None
        ),
        "mean_rounds_failure": (
            sum(int(row.get("num_chunks", 0)) for row in rows if not row.get("success")) / (len(rows) - successes)
            if len(rows) > successes else None
        ),
        "end_reasons": dict(sorted(Counter(row.get("end_reason", "missing") for row in rows).items())),
    }


def compare_paired(a: dict, b: dict) -> dict:
    keys = sorted(set(a) & set(b))
    a_success_b_failure = sum(bool(a[key].get("success")) and not bool(b[key].get("success")) for key in keys)
    a_failure_b_success = sum(not bool(a[key].get("success")) and bool(b[key].get("success")) for key in keys)
    a_rate = sum(bool(a[key].get("success")) for key in keys) / len(keys) if keys else math.nan
    b_rate = sum(bool(b[key].get("success")) for key in keys) / len(keys) if keys else math.nan
    return {
        "paired_episodes": len(keys),
        "a_success_b_failure": a_success_b_failure,
        "a_failure_b_success": a_failure_b_success,
        "paired_success_rate_difference_a_minus_b": a_rate - b_rate,
        "mcnemar_exact_p": exact_mcnemar(a_success_b_failure, a_failure_b_success),
    }


def parse_run(value: str) -> tuple[str, list[Path]]:
    label, separator, paths = value.partition("=")
    if not separator or not label or not paths:
        raise argparse.ArgumentTypeError("run must be LABEL=PATH[,PATH...]")
    return label, [Path(path) for path in paths.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, type=parse_run)
    parser.add_argument("--compare", action="append", default=[], help="paired comparison as LABEL_A,LABEL_B")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    records = {label: load_runs(paths) for label, paths in args.run}
    report = {"groups": {label: summarize(rows) for label, rows in records.items()}, "paired_comparisons": {}}
    for comparison in args.compare:
        a_label, separator, b_label = comparison.partition(",")
        if not separator or a_label not in records or b_label not in records:
            raise SystemExit(f"invalid comparison: {comparison}")
        report["paired_comparisons"][comparison] = compare_paired(records[a_label], records[b_label])
    rendered = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
