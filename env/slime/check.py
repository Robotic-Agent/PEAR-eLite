"""Smoke check of the slime environment on one GPU: imports, the GPU architectures the builds cover, and one small
forward pass of each CUDA extension the Qwen3.5 path uses. Prints CHECK_OK when every check passes.
usage: source $PREFIX/activate.sh && python env/slime/check.py
"""
import importlib
import traceback

import torch

results = []


def check(name, fn):
    try:
        info = fn()
        results.append((name, True, info))
    except Exception as e:  # noqa: BLE001 - every failure is reported
        results.append((name, False, f"{type(e).__name__}: {e}"))
        traceback.print_exc()


def versions():
    out = {}
    for m in ("torch", "sglang", "sgl_kernel", "flashinfer", "transformers", "transformer_engine", "megatron.core",
              "fla", "mamba_ssm", "causal_conv1d", "torch_memory_saver", "slime", "apex", "ray"):
        mod = importlib.import_module(m)
        out[m] = getattr(mod, "__version__", "ok")
    return out


def gpu():
    cap = torch.cuda.get_device_capability()
    return f"{torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]}, torch archs {torch.cuda.get_arch_list()}"


def te_linear():
    import transformer_engine.pytorch as te

    layer = te.Linear(256, 256, params_dtype=torch.bfloat16).cuda()
    x = torch.randn(32, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    layer(x).sum().backward()
    return "forward/backward"


def te_attention():
    import transformer_engine.pytorch as te

    attn = te.DotProductAttention(8, 128, attn_mask_type="causal", qkv_format="bshd").cuda()
    q, k, v = (torch.randn(1, 512, 8, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    return f"out {tuple(attn(q, k, v).shape)}"


def causal_conv():
    from causal_conv1d import causal_conv1d_fn

    x = torch.randn(1, 64, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(64, 4, device="cuda", dtype=torch.bfloat16)
    return f"out {tuple(causal_conv1d_fn(x, w).shape)}"


def gated_delta():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    b, t, h, d = 1, 256, 4, 64
    q, k, v = (torch.randn(b, t, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    k = torch.nn.functional.normalize(k, dim=-1)
    g = -torch.rand(b, t, h, device="cuda", dtype=torch.float32)
    beta = torch.rand(b, t, h, device="cuda", dtype=torch.bfloat16)
    o, _ = chunk_gated_delta_rule(q, k, v, g, beta)
    return f"out {tuple(o.shape)}"


def apex_norm():
    from apex.normalization import FusedRMSNorm

    n = FusedRMSNorm(256).cuda()
    return f"out {tuple(n(torch.randn(8, 256, device='cuda')).shape)}"


def sgl_kernel_op():
    import sgl_kernel

    x = torch.randn(16, 512, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(16, 256, device="cuda", dtype=torch.bfloat16)
    sgl_kernel.silu_and_mul(x, out)
    return "silu_and_mul"


def megatron_slime():
    importlib.import_module("megatron.training")
    importlib.import_module("slime_plugins.models.qwen3_5")
    return "megatron.training, slime_plugins.models.qwen3_5"


check("versions", versions)
check("gpu", gpu)
check("transformer_engine linear", te_linear)
check("transformer_engine attention", te_attention)
check("causal_conv1d", causal_conv)
check("fla gated delta rule", gated_delta)
check("apex fused rmsnorm", apex_norm)
check("sgl_kernel", sgl_kernel_op)
check("megatron + slime qwen3_5", megatron_slime)

for name, ok, info in results:
    print(f"{'OK  ' if ok else 'FAIL'} {name}: {info}")
print("CHECK_OK" if all(ok for _, ok, _ in results) else "CHECK_FAILED")
