"""Compare two teachers' top-k prompt log-probs (outputs of teacher_prompt_logprobs.py) position by position.
usage: python compare_teacher_logprobs.py <reference json> <candidate json>
"""
import json
import math
import sys

ref, cand = (json.load(open(p)) for p in sys.argv[1:3])
n = top1 = 0
d_actual, kl_terms, overlap = [], [], []
for a, b in zip(ref, cand, strict=True):
    for x, y in zip(a, b, strict=True):
        assert x["_tok"] == y["_tok"]
        t = str(x["_tok"])
        xa = {k: v for k, v in x.items() if k != "_tok"}
        ya = {k: v for k, v in y.items() if k != "_tok"}
        n += 1
        top1 += max(xa, key=xa.get) == max(ya, key=ya.get)
        if t in xa and t in ya:
            d_actual.append(abs(xa[t] - ya[t]))
        # KL(ref || cand) over the reference's top-k, candidate's missing entries floored at its k-th log-prob
        floor = min(ya.values())
        kl_terms.append(sum(math.exp(lp) * (lp - ya.get(k, floor)) for k, lp in xa.items()))
        overlap.append(len(set(xa) & set(ya)) / len(xa))

d_actual.sort()
print(f"positions {n}")
print(f"top-1 agreement {top1 / n:.4f}")
print(f"top-k overlap mean {sum(overlap) / n:.4f}")
print(f"|dlogp| of the actual token: mean {sum(d_actual) / len(d_actual):.4f}, "
      f"p50 {d_actual[len(d_actual) // 2]:.4f}, p99 {d_actual[int(len(d_actual) * 0.99)]:.4f}")
print(f"KL(ref||cand) over ref top-k: mean {sum(kl_terms) / n:.5f}")
