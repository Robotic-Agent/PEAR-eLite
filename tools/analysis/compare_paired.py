"""Paired comparison of two LIBERO eval runs on their common episodes (suite, task, trial).
usage: python compare_paired.py <label> <new run dir> <base run dir>[,<base run dir>...]
"""
import json
import math
import sys
from pathlib import Path


def load(dirs):
    out = {}
    for d in dirs.split(","):
        p = Path(d) / "episodes.jsonl"
        if not p.exists():
            continue
        for line in p.open():
            e = json.loads(line)
            if not e.get("done", True):
                continue
            k = (e["env"]["suite"], e["env"]["task_id"], e["env"]["trial_id"])
            out[k] = e
    return out


def per_turn(eps):
    t = [x for e in eps for x in e.get("turn_tokens", [])]
    return sum(t) / len(t) if t else float("nan")


def binom_two_sided(b, c):
    # exact McNemar: P(X <= min(b,c)) * 2 for X ~ Bin(b + c, 0.5)
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


label, new_dir, base_dirs = sys.argv[1:4]
new, base = load(new_dir), load(base_dirs)
keys = sorted(set(new) & set(base))
nn = [new[k] for k in keys]
bb = [base[k] for k in keys]
sn, sb = sum(e["success"] for e in nn), sum(e["success"] for e in bb)
b = sum(1 for x, y in zip(nn, bb) if x["success"] and not y["success"])
c = sum(1 for x, y in zip(nn, bb) if y["success"] and not x["success"])
print(f"{label}: new episodes {len(new)} (success {sum(e['success'] for e in new.values())}); paired {len(keys)}")
if keys:
    print(f"  paired success new {sn}/{len(keys)} = {sn/len(keys):.1%}, base {sb}/{len(keys)} = {sb/len(keys):.1%}; "
          f"new-only {b}, base-only {c}, McNemar exact p = {binom_two_sided(b, c):.3f}")
    print(f"  tokens per turn new {per_turn(nn):.0f}, base {per_turn(bb):.0f}; "
          f"turns per episode new {sum(e['num_chunks'] for e in nn)/len(nn):.1f}, "
          f"base {sum(e['num_chunks'] for e in bb)/len(bb):.1f}")
