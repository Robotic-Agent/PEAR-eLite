"""Ulysses SDPA vs single-device SDPA on packed sequences: outputs and gradients.
usage (verl environment, 3rdparty/verl/src on the path):
  torchrun --nproc_per_node=<sp> tests/gpu/test_ulysses_sdpa.py   # prints ULYSSES_SDPA_OK
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F

from verl.models.transformers.ulysses_sdpa import ulysses_sdpa_attention_forward  # noqa: E402

from verl.utils.ulysses import set_ulysses_sequence_parallel_group  # noqa: E402

dist.init_process_group("nccl")
rank, sp = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)
set_ulysses_sequence_parallel_group(dist.group.WORLD)


class M(torch.nn.Module):
    num_key_value_groups = 4
    is_causal = True


torch.manual_seed(0)
hq, hk, d = 16, 4, 256
lens = [1000, 3000, 600, 1544]  # packed sequences; total divisible by sp
S = sum(lens)
pos = torch.cat([torch.arange(n) for n in lens]).view(1, S).cuda()
q = torch.randn(1, hq, S, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(1, hk, S, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(1, hk, S, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
go = torch.randn(1, S, hq, d, device="cuda", dtype=torch.bfloat16)
scale = d ** -0.5

# reference: whole sequence on one device, causal per packed sequence
ref = []
start = 0
for n in lens:
    sl = slice(start, start + n)
    kk = k[:, :, sl].repeat_interleave(hq // hk, dim=1)
    vv = v[:, :, sl].repeat_interleave(hq // hk, dim=1)
    ref.append(F.scaled_dot_product_attention(q[:, :, sl], kk, vv, is_causal=True, scale=scale))
    start += n
ref = torch.cat(ref, dim=2).transpose(1, 2)  # (1, S, hq, d)
(ref.float() * go.float()).sum().backward()
gq, gk, gv = q.grad.clone(), k.grad.clone(), v.grad.clone()

# Ulysses: each rank holds S/sp tokens
c = S // sp
sl = slice(rank * c, (rank + 1) * c)
ql, kl, vl = (x.detach()[:, :, sl].clone().requires_grad_(True) for x in (q, k, v))
out, _ = ulysses_sdpa_attention_forward(M(), ql, kl, vl, None, scaling=scale, position_ids=pos[:, sl].contiguous())
(out.float() * go[:, sl].float()).sum().backward()


def err(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


e = [err(out, ref[:, sl]), err(ql.grad, gq[:, :, sl]), err(kl.grad, gk[:, :, sl]), err(vl.grad, gv[:, :, sl])]
# what verl does today under SDPA: attention within the local slice only
local_pos = pos[:, sl]
starts = [i for i in range(c) if int(local_pos[0, i]) == 0] + [c]
if starts[0] != 0:
    starts = [0] + starts
bad = []
for a, b in zip(starts[:-1], starts[1:]):
    kk = kl.detach()[:, :, a:b].repeat_interleave(hq // hk, dim=1)
    vv = vl.detach()[:, :, a:b].repeat_interleave(hq // hk, dim=1)
    bad.append(F.scaled_dot_product_attention(ql.detach()[:, :, a:b], kk, vv, is_causal=True, scale=scale))
bad = torch.cat(bad, dim=2).transpose(1, 2)
print(f"rank {rank}: rel max err out {e[0]:.2e} dq {e[1]:.2e} dk {e[2]:.2e} dv {e[3]:.2e}; "
      f"local-only attention err {err(bad, ref[:, sl]):.2e}", flush=True)
ok = all(x < 2e-2 for x in e)
t = torch.tensor([int(ok)], device="cuda")
dist.all_reduce(t, op=dist.ReduceOp.MIN)
if rank == 0:
    print("ULYSSES_SDPA_OK" if t.item() else "ULYSSES_SDPA_MISMATCH", flush=True)
dist.destroy_process_group()
