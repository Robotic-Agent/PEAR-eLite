"""Per-episode chain-of-thought length and duration of a pear_libero_eval run: finished episodes and those in flight."""
import glob
import json
import os
import sys
import time

run = sys.argv[1]
os.chdir(run)
rows = [json.loads(l) for l in open("episodes.jsonl")]
ok = [r for r in rows if r["end_reason"] != "error"]
print("finished (non-error):", len(ok))
for r in ok[-25:]:
    e, tt = r["env"], r["turn_tokens"]
    print(f"{e['suite'][7:]:8s} t{e['task_id']} r{e['trial_id']} succ={int(r['success'])} end={r['end_reason']:6s} "
          f"rounds={r['num_chunks']:2d} gen={r['generated_tokens']:6d} ctx={r['response_tokens']:6d} "
          f"gen/round={r['generated_tokens'] // max(len(tt), 1):5d} max_round={max(tt) if tt else 0:5d} "
          f"min={r['seconds'] / 60:5.1f}")
if ok:
    n = len(ok)
    print(f"mean: rounds={sum(r['num_chunks'] for r in ok) / n:.1f} gen={sum(r['generated_tokens'] for r in ok) / n:.0f} "
          f"gen/round={sum(r['generated_tokens'] for r in ok) / max(sum(len(r['turn_tokens']) for r in ok), 1):.0f} "
          f"minutes={sum(r['seconds'] for r in ok) / n / 60:.1f}")
now = time.time()
done = {(r["env"]["suite"], r["env"]["task_id"], r["env"]["trial_id"]) for r in ok}
print("--- in flight (turn log written in the last hour, episode not finished)")
for f in sorted(glob.glob("turns/*.jsonl"), key=os.path.getmtime):
    stem = os.path.basename(f)[:-6]
    suite, task, trial = stem.rsplit("_", 2)
    if (suite, int(task), int(trial)) in done or now - os.path.getmtime(f) > 3600:
        continue
    turns = [json.loads(l) for l in open(f)]
    g = [x["gen"] for x in turns]
    print(f"{stem:22s} rounds={len(turns):2d} gen={sum(g):6d} gen/round={sum(g) // max(len(g), 1):5d} "
          f"max_round={max(g) if g else 0:5d} ctx_now={turns[-1]['prompt_tokens'] if turns else 0:6d} "
          f"running_min={(now - os.path.getctime(f)) / 60:5.1f}")
