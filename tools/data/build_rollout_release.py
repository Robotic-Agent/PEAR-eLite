"""Pack the 27B PEAR rollouts under ROLLOUTS_DIR into a ModelScope dataset: one parquet row per episode, per benchmark.

The latest record per environment key (suite, category/perturbation, task, trial) wins; a resumed shard can rerun one.
Columns: episode metadata, rounds (full model response with its thinking, the parsed action, the robot state before and
after, the raw round record as JSON), the two camera PNGs per round, the episode's GIF and its native actions.
usage: ROLLOUTS_DIR=<rollout runs> python tools/data/build_rollout_release.py OUT_DIR
"""

import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.environ["ROLLOUTS_DIR"]  # the directory holding the rollout runs (<run>/episodes.jsonl)
CONFIGS = {
    "libero": "rollout_libero_s*of8",
    "libero_plus": "rollout_libero_plus_s*of16",
    "libero_pro": "rollout_libero_pro_s*of16",
}
SHARD_EPISODES = 100


def records(pattern):
    latest = {}
    for d in sorted(glob.glob(f"{ROOT}/{pattern}")):
        f = f"{d}/episodes.jsonl"
        if not os.path.exists(f):
            continue
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest[json.dumps(r["env"], sort_keys=True)] = (d, r)
    return list(latest.values())


def read(path):
    with open(path, "rb") as f:
        return f.read()


def build(item):
    d, r = item
    name = "_".join(str(v) for v in r["env"].values())
    e = f"{d}/episodes/{name}"
    try:
        t = json.load(open(f"{e}/trajectory.json"))
    except (OSError, ValueError):
        return None, "no_trajectory"
    s = t["summary"]
    texts = {}
    tf = f"{d}/turns/{name}.jsonl"
    if os.path.exists(tf):
        for line in open(tf):
            try:
                x = json.loads(line)
            except json.JSONDecodeError:
                continue
            texts[int(x["turn"])] = x.get("text")
    rounds, missing_text, missing_frame = [], 0, 0
    for rd in t["rounds"]:
        i = int(rd["round"])
        text = texts.get(i)
        missing_text += text is None
        frames = {}
        for cam in ("agentview", "wrist"):
            p = f"{e}/frames/round_{i:03d}_{cam}.png"
            frames[cam] = read(p) if os.path.exists(p) else None
            missing_frame += frames[cam] is None
        rounds.append({
            "round": i,
            "response": text,
            "answer": rd.get("answer"),
            "action": json.dumps(rd.get("action")),
            "state": json.dumps(rd.get("state")),
            "state_after": json.dumps(rd.get("state_after")),
            "outcome": rd.get("outcome"),
            "finish": rd.get("finish"),
            "prompt_tokens": rd.get("prompt_tokens"),
            "gen_tokens": rd.get("gen_tokens"),
            "agentview_png": frames["agentview"],
            "wrist_png": frames["wrist"],
            "record": json.dumps(rd),
        })
    env = s["env"]
    gif = f"{e}/rounds.gif"
    row = {
        "uid": s.get("uid"),
        "suite": env["suite"],
        "category": env.get("category") or env.get("perturbation"),
        "task_id": env["task_id"],
        "trial_id": env["trial_id"],
        "instruction": t.get("instruction"),
        "success": bool(s.get("success")),
        "end_reason": s.get("end_reason"),
        "num_rounds": len(rounds),
        "executed_steps": s.get("executed_steps"),
        "response_tokens": s.get("response_tokens"),
        "settings": json.dumps(t.get("settings")),
        "rounds": rounds,
        "native_actions": t.get("native_actions"),
        "video_gif": read(gif) if os.path.exists(gif) else None,
    }
    flags = "ok" if not (missing_text or missing_frame) else f"missing_text={missing_text},missing_frame={missing_frame}"
    return row, flags


def main(out):
    report = {}
    with ProcessPoolExecutor(64) as pool:
        for config, pattern in CONFIGS.items():
            items = sorted(records(pattern), key=lambda x: json.dumps(x[1]["env"], sort_keys=True))
            items = items[: int(os.environ.get("LIMIT", "0")) or None]
            os.makedirs(f"{out}/data/{config}", exist_ok=True)
            stats = {"episodes": 0, "success": 0, "rounds": 0, "skipped": {}, "incomplete": 0, "by_suite": {}}
            buf, shard = [], 0

            def flush():
                nonlocal buf, shard
                if buf:
                    pq.write_table(pa.Table.from_pylist(buf), f"{out}/data/{config}/train-{shard:05d}.parquet",
                                   row_group_size=10, compression="zstd")
                    shard += 1
                    buf = []

            for row, flags in pool.map(build, items, chunksize=4):
                if row is None:
                    stats["skipped"][flags] = stats["skipped"].get(flags, 0) + 1
                    continue
                stats["incomplete"] += flags != "ok"
                stats["episodes"] += 1
                stats["success"] += row["success"]
                stats["rounds"] += row["num_rounds"]
                b = stats["by_suite"].setdefault(row["suite"], [0, 0])
                b[0] += 1
                b[1] += row["success"]
                buf.append(row)
                if len(buf) >= SHARD_EPISODES:
                    flush()
            flush()
            stats["shards"] = shard
            report[config] = stats
            print(config, json.dumps(stats), flush=True)
    json.dump(report, open(f"{out}/stats.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
