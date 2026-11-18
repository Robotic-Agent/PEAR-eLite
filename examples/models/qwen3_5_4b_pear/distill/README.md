# PEAR-27B to Qwen3.5-4B distillation

`run_opd_5090.sh` runs one-node on-policy distillation on 8 RTX 5090 GPUs. Its default smoke configuration uses two
LIBERO episodes, generates one student trajectory per episode, obtains the PEAR-27B teacher's top-32 distribution for
every student response token, performs one direct k3 distillation update and writes a checkpoint.

The GPU allocation is fixed to disjoint pools during an update:

| Pool | GPUs | Parallelism |
| --- | ---: | --- |
| PEAR-27B teacher | 4 | TP4 |
| Qwen3.5-4B actor | 2 | FSDP2 |
| Qwen3.5-4B rollout | 2 | two TP1 replicas |

The PEAR optimized actor and rollout kernels support datacenter Blackwell and Hopper. RTX 5090 is SM120, so this
recipe uses the generic Transformers SDPA actor and generic vLLM rollout paths. The smoke test establishes functional
correctness and measured memory use before increasing response length, batch size or training steps.

Start Ray on the allocation, then launch the recipe:

```bash
export WORK_DIR=<the work dir that scripts/install.sh built>
LIBERO_SERVERS_PER_NODE=2 bash scripts/start_ray_node.sh trainer_rollout &
RAY_ADDRESS=auto RUN=<run dir> \
  bash examples/models/qwen3_5_4b_pear/distill/run_opd_5090.sh
```

The student directory must use the PEAR-27B tokenizer and chat template while retaining the Qwen3.5-4B weights. The
teacher and student vocabularies therefore use the same token ids during teacher scoring.
