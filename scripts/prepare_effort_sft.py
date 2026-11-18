#!/usr/bin/env python3
"""Build multi-turn multimodal SFT parquet files from successful PEAR evaluation trajectories."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REMINDER = "The current camera frames follow. End your response with one json block containing the next action."
CAMERAS = ("agentview", "wrist")
# trial ids left out of the data (--exclude-trials), e.g. the initial states of the evaluation episodes
EXCLUDE_TRIALS: set[int] = set()
# successful episodes kept per task (the shortest first)
PER_TASK = 1


@dataclass(frozen=True)
class Source:
    weight: int
    paths: tuple[Path, ...]


def parse_source(value: str) -> Source:
    weight_text, separator, path_text = value.partition(":")
    if not separator or not weight_text.isdigit() or int(weight_text) < 1 or not path_text:
        raise argparse.ArgumentTypeError("source must be WEIGHT:PATH[,PATH...] with a positive integer weight")
    paths = tuple(Path(path) for path in path_text.split(",") if path)
    if not paths:
        raise argparse.ArgumentTypeError("source must contain at least one path")
    return Source(int(weight_text), paths)


def successful_episodes(run: Path) -> Iterable[dict]:
    """Yield the final successful record for every episode in a resumable run."""
    records = run / "episodes.jsonl"
    if not records.is_file():
        raise FileNotFoundError(records)
    latest: dict[tuple[str, int, int], dict] = {}
    latest_non_error: dict[tuple[str, int, int], dict] = {}
    for line in records.read_text().splitlines():
        record = json.loads(line)
        env = record["env"]
        key = (env["suite"], int(env["task_id"]), int(env["trial_id"]))
        latest[key] = record
        if record.get("end_reason") != "error":
            latest_non_error[key] = record
    for key, record in latest.items():
        selected = latest_non_error.get(key, record)
        if selected.get("success") is True and key[2] not in EXCLUDE_TRIALS:
            yield selected


def select_shortest_per_task(
    runs: Path | Iterable[Path], max_response_tokens: int = 0, all_successes: bool = False
) -> list[tuple[Path, dict]]:
    if isinstance(runs, Path):
        runs = (runs,)
    if all_successes:
        kept = []
        for run in runs:
            for record in successful_episodes(run):
                response_tokens = int(record.get("response_tokens", record["generated_tokens"]))
                if max_response_tokens and response_tokens > max_response_tokens:
                    continue
                kept.append((run, record))
        env_key = lambda item: (item[1]["env"]["suite"], int(item[1]["env"]["task_id"]), int(item[1]["env"]["trial_id"]))
        return sorted(kept, key=env_key)
    by_task: dict[tuple[str, int], list[tuple[Path, dict]]] = {}
    for run in runs:
        for record in successful_episodes(run):
            response_tokens = int(record.get("response_tokens", record["generated_tokens"]))
            if max_response_tokens and response_tokens > max_response_tokens:
                continue
            env = record["env"]
            by_task.setdefault((env["suite"], int(env["task_id"])), []).append((run, record))
    selected = []
    for key in sorted(by_task):
        # the PER_TASK shortest successes of the task (fewest generated tokens; trial id breaks ties)
        ranked = sorted(by_task[key], key=lambda item: (int(item[1]["generated_tokens"]), int(item[1]["env"]["trial_id"])))
        selected.extend(ranked[:PER_TASK])
    return selected


def episode_name(record: dict) -> str:
    env = record["env"]
    return f"{env['suite']}_{env['task_id']}_{env['trial_id']}"


def build_row(run: Path, record: dict, source_name: str) -> dict:
    """Reconstruct the exact observation/assistant alternation saved by pear_libero_eval.py."""
    from pear.agent_loop.protocol import JsonBlockProtocol

    root = Path(__file__).resolve().parents[1]
    system_template = (root / "pear/envs/system_prompt.md").read_text()
    environment_prompt = (root / "pear/envs/libero/environment_prompt.md").read_text().strip()

    name = episode_name(record)
    episode_dir = run / "episodes" / name
    trajectory = json.loads((episode_dir / "trajectory.json").read_text())
    turn_path = run / "turns" / f"{name}.jsonl"
    turns = [json.loads(line) for line in turn_path.read_text().splitlines()]
    rounds = trajectory["rounds"]
    if len(turns) != len(rounds):
        raise ValueError(f"{name}: {len(turns)} saved turns but {len(rounds)} trajectory rounds")

    messages = [{
        "role": "system",
        "content": JsonBlockProtocol.system_prompt(system_template, environment_prompt, trajectory["instruction"]),
    }]
    images = []
    for index, (turn, round_record) in enumerate(zip(turns, rounds)):
        user_text = JsonBlockProtocol.turn_text(round_record["state"], REMINDER)
        user_text += "\n[agentview]<image>\n[wrist]<image>"
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": turn["text"]})
        for camera in CAMERAS:
            frame = episode_dir / "frames" / f"round_{index:03d}_{camera}.png"
            images.append({"bytes": frame.read_bytes()})

    return {
        "messages": messages,
        "images": images,
        "enable_thinking": True,
        "source": source_name,
        "suite": record["env"]["suite"],
        "task_id": int(record["env"]["task_id"]),
        "trial_id": int(record["env"]["trial_id"]),
        "generated_tokens": int(record["generated_tokens"]),
        "rounds": int(record["num_chunks"]),
    }


def build_rows(sources: list[Source], max_response_tokens: int = 0, all_successes: bool = False) -> tuple[list[dict], dict]:
    rows = []
    report = {
        "sources": [],
        "rows": 0,
        "unique_episodes": 0,
        "max_response_tokens": max_response_tokens,
        "all_successes": all_successes,
    }
    for source_index, source in enumerate(sources):
        selected = select_shortest_per_task(source.paths, max_response_tokens, all_successes)
        source_name = source.paths[0].parent.name or source.paths[0].name or f"source_{source_index}"
        source_rows, skipped = [], []
        for run, record in selected:
            # a run still being written (a rollout synced mid-episode) can have an incomplete episode: skip it
            try:
                source_rows.append(build_row(run, record, source_name))
            except (OSError, ValueError, KeyError) as error:
                skipped.append(f"{run.name}/{episode_name(record)}: {type(error).__name__}: {str(error)[:120]}")
        report.setdefault("skipped", []).extend(skipped)
        for repeat in range(source.weight):
            for row in source_rows:
                rows.append({**row, "repeat": repeat})
        report["sources"].append({
            "paths": [str(path) for path in source.paths],
            "weight": source.weight,
            "successful_tasks": len(source_rows),
            "weighted_rows": len(source_rows) * source.weight,
        })
        report["unique_episodes"] += len(source_rows)
    report["rows"] = len(rows)
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", action="append", required=True, type=parse_source,
        help="WEIGHT:PATH[,PATH...]; retain the shortest successful trial per task across that source's paths",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--max-response-tokens",
        type=int,
        default=0,
        help="skip an episode whose cumulative response context exceeds this limit; 0 disables the filter",
    )
    parser.add_argument(
        "--all-successes",
        action="store_true",
        help="keep every successful episode instead of the shortest successful trial per task",
    )
    parser.add_argument(
        "--exclude-trials",
        default="",
        help="comma-separated trial ids to leave out (the evaluation's initial states, e.g. 0 for libero_sub1.json)",
    )
    parser.add_argument("--per-task", type=int, default=1, help="keep the K shortest successful episodes of each task")
    args = parser.parse_args()
    global PER_TASK
    PER_TASK = args.per_task
    EXCLUDE_TRIALS.update(int(x) for x in args.exclude_trials.split(",") if x.strip())

    if args.max_response_tokens < 0:
        parser.error("--max-response-tokens must be non-negative")
    rows, report = build_rows(args.source, args.max_response_tokens, args.all_successes)
    if not rows:
        raise SystemExit("no successful trajectories were found")
    import pandas as pd

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out, index=False, row_group_size=16)  # nested image columns: keep each row group under the 2 GiB Arrow array limit
    report_path = args.out.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
