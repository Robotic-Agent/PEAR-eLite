"""LIBERO evaluation of a PEAR checkpoint without verl/Ray: vLLM's AsyncLLM in this process (tensor parallel), LIBERO
in worker processes of PEAR's LIBERO environment. The conversation follows PEAR's agent loop (pear/agent_loop,
pear/envs/tool.py): PEAR's system prompt with LIBERO's ENVIRONMENT section, one user turn per observation (the state
JSON, the reminder, "[agentview]" + frame, "[wrist]" + frame), the model's whole generation (its thinking included)
kept in the context, the last ```json block after </think> as the action, and an episode budget of response tokens
(everything after the first prompt). Results: one JSON line per episode in --out/episodes.jsonl (resumable).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

from PIL import Image

PEAR = Path(os.environ.get("PEAR_REPO", Path(__file__).resolve().parents[2]))
WORK = os.environ.get("WORK_DIR", "")
sys.path.insert(0, str(PEAR))
from pear.agent_loop.protocol import InvalidAction, JsonBlockProtocol  # noqa: E402  (no verl import)

SYSTEM_TEMPLATE = (PEAR / "pear/envs/system_prompt.md").read_text()
ENVIRONMENT_PROMPT = (PEAR / "pear/envs/libero/environment_prompt.md").read_text().strip()
REMINDER = "The current camera frames follow. End your response with one json block containing the next action."
CAMERAS = ("agentview", "wrist")
SCHEMA = None  # MotionController.SCHEMA, read from the worker's first reply (its state's parts and quantities)
THINK_END = "</think>"


def parse_range(spec) -> list[int]:
    if isinstance(spec, int):
        return [spec]
    out = []
    for part in str(spec).split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def episodes_of(config_path: str) -> list[dict]:
    cfg = json.loads(Path(config_path).read_text())
    env = cfg["env"]
    suites = env["suite"] if isinstance(env["suite"], list) else [env["suite"]]
    return [
        {"suite": s, "task_id": t, "trial_id": r}
        for s in suites
        for t in parse_range(env["task_id"])
        for r in parse_range(env["trial_id"])
    ]


class Worker:
    """One LIBERO simulator process (libero_worker.py) in the LIBERO environment's Python."""

    def __init__(self, python: str, script: str, mesa: str, libero_config: str, log: Path) -> None:
        variables = {
            **os.environ,
            "PYTHONPATH": str(PEAR),
            "LIBERO_CONFIG_PATH": libero_config,
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
            "LD_LIBRARY_PATH": f"{mesa}/lib",
            "__EGL_VENDOR_LIBRARY_FILENAMES": f"{mesa}/share/glvnd/egl_vendor.d/50_mesa.json",
            "LIBGL_DRIVERS_PATH": f"{mesa}/lib/dri",
            "GALLIUM_DRIVER": "llvmpipe",
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "LP_NUM_THREADS": "4",
            "XDG_RUNTIME_DIR": f"/tmp/xdg_{os.environ.get('USER', 'root')}",
            "CUDA_VISIBLE_DEVICES": "",
            "USER": os.environ.get("USER", "root"),
        }
        self.args = ([python, "-u", script], variables, log)
        self.proc = None
        self.lock = asyncio.Lock()

    async def start(self) -> None:
        cmd, variables, log = self.args
        self.proc = await asyncio.create_subprocess_exec(
            *cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(log, "ab"), env=variables,
            limit=64 * 1024 * 1024,
        )
        ready = json.loads(await asyncio.wait_for(self.proc.stdout.readline(), 900))
        assert ready.get("ready"), ready

    async def call(self, env_args: dict, history: list, action) -> dict:
        async with self.lock:
            for attempt in range(3):
                try:
                    if self.proc is None or self.proc.returncode is not None:
                        await self.start()
                    msg = json.dumps({"env_args": env_args, "history": history, "action": action}) + "\n"
                    self.proc.stdin.write(msg.encode())
                    await self.proc.stdin.drain()
                    reply = json.loads(await asyncio.wait_for(self.proc.stdout.readline(), 600))
                    if "error" in reply:
                        raise RuntimeError(reply["error"])
                    return reply["result"]
                except (asyncio.TimeoutError, json.JSONDecodeError, BrokenPipeError, ConnectionResetError) as exc:
                    print(f"worker failure ({exc!r}), restart {attempt + 1}", flush=True)
                    if self.proc and self.proc.returncode is None:
                        self.proc.kill()
                    self.proc = None
            raise RuntimeError("simulator worker failed 3 times")


class Evaluator:
    def __init__(self, args) -> None:
        self.args = args
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "turns").mkdir(exist_ok=True)
        self.results_path = self.out / "episodes.jsonl"
        self.workers: asyncio.Queue = asyncio.Queue()
        self.tokens_generated = 0
        self.t0 = time.time()
        self.engine_dead = False

    def build_engine(self):
        from transformers import AutoProcessor
        from vllm import SamplingParams  # noqa: F401
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        a = self.args
        self.processor = AutoProcessor.from_pretrained(a.model)
        self.tokenizer = self.processor.tokenizer
        engine_args = AsyncEngineArgs(
            model=a.model,
            tensor_parallel_size=a.tp,
            max_model_len=a.max_prompt + a.max_response,
            gpu_memory_utilization=a.gpu_mem,
            max_num_seqs=a.sessions,
            enable_prefix_caching=True,
            limit_mm_per_prompt={"image": {"count": (a.max_prompt + a.max_response) // 256, "width": 512, "height": 512}},
            kv_cache_dtype=a.kv_cache_dtype,
            seed=a.seed,
            **json.loads(a.engine_kwargs),
        )
        self.engine = AsyncLLM.from_engine_args(engine_args)

    def render(self, messages: list[dict], add_generation_prompt: bool) -> str:
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=True,
            reasoning_effort=self.args.effort,
        )

    def user_content(self, result: dict) -> tuple[list[dict], list[Image.Image]]:
        blocks = [{"type": "text", "text": JsonBlockProtocol.turn_text(result["state"], REMINDER)}]
        images = []
        for name, frame in zip(CAMERAS, result["images"]):
            image = Image.open(io.BytesIO(base64.b64decode(frame))).convert("RGB")
            blocks.append({"type": "text", "text": f"[{name}]"})
            blocks.append({"type": "image", "image": image})
            images.append(image)
        return blocks, images

    async def generate(self, prompt: str, images: list, max_tokens: int, request_id: str):
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=self.args.temperature, top_p=self.args.top_p, top_k=-1, max_tokens=max_tokens,
            skip_special_tokens=False, spaces_between_special_tokens=False,
        )
        final = None
        async for output in self.engine.generate(
            {"prompt": prompt, "multi_modal_data": {"image": images}}, params, request_id=request_id
        ):
            final = output
        return final

    @staticmethod
    def save_frames(frames_dir: Path, index: int, result: dict) -> None:
        """Writes the observation's frames (the PNGs the model sees) as round_<index>_<camera>.png."""
        if result.get("images") is None:
            return
        for name, frame in zip(CAMERAS, result["images"]):
            (frames_dir / f"round_{index:03d}_{name}.png").write_bytes(base64.b64decode(frame))

    @staticmethod
    def write_gif(frames_dir: Path, path: Path, labels: list[str]) -> None:
        """One frame per observation: agentview | wrist side by side, the round's label on top."""
        from PIL import ImageDraw

        frames = []
        for i, label in enumerate(labels):
            views = [frames_dir / f"round_{i:03d}_{name}.png" for name in CAMERAS]
            if not all(v.exists() for v in views):
                continue
            images = [Image.open(v).convert("RGB").resize((256, 256)) for v in views]
            canvas = Image.new("RGB", (512, 276), "white")
            for j, image in enumerate(images):
                canvas.paste(image, (256 * j, 20))
            ImageDraw.Draw(canvas).text((4, 4), label[:90], fill="black")
            frames.append(canvas)
        if frames:
            frames[0].save(path, save_all=True, append_images=frames[1:], duration=800, loop=0)

    async def episode(self, env_args: dict) -> dict:
        a = self.args
        worker = await self.workers.get()
        uid = uuid.uuid4().hex[:12]
        name = f"{env_args['suite']}_{env_args['task_id']}_{env_args['trial_id']}"
        turn_log = open(self.out / "turns" / f"{name}.jsonl", "w")
        # the episode's recording: the frames of every observation, its rounds and its native actions
        ep_dir = self.out / "episodes" / name
        frames_dir = ep_dir / "frames"
        if ep_dir.exists():
            import shutil

            shutil.rmtree(ep_dir)
        frames_dir.mkdir(parents=True)
        rounds: list[dict] = []
        record = {"env": env_args, "uid": uid, "success": False, "done": False, "end_reason": None, "num_chunks": 0,
                  "executed_steps": 0, "response_tokens": 0, "generated_tokens": 0, "turn_tokens": []}
        start = time.time()
        history: list = []
        try:
            result = await worker.call(env_args, history, None)
            self.save_frames(frames_dir, 0, result)
            system = JsonBlockProtocol.system_prompt(
                SYSTEM_TEMPLATE, f"{ENVIRONMENT_PROMPT}" if ENVIRONMENT_PROMPT else "", result["instruction"]
            )
            content, images = self.user_content(result)
            messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
            prompt = self.render(messages, True)
            # the system part of a render, to cut later turns' user blocks out of renders of [system, user]
            system_render = prompt[: prompt.index("<|im_start|>user")]
            example = self.out / "prompt_example.txt"
            if not example.exists():
                example.write_text(prompt)
            first_prompt_tokens = None
            all_images = list(images)
            record["instruction"] = result["instruction"]
            while True:
                used = record["response_tokens"]
                max_tokens = a.max_response - used
                if max_tokens <= 0:
                    record["end_reason"] = "budget"
                    break
                t_gen = time.time()
                out = await self.generate(prompt, all_images, max_tokens, f"{uid}-{record['num_chunks']}")
                t_gen = time.time() - t_gen
                completion = out.outputs[0]
                text = completion.text
                n_prompt = len(out.prompt_token_ids)
                if first_prompt_tokens is None:
                    first_prompt_tokens = n_prompt
                n_gen = len(completion.token_ids)
                self.tokens_generated += n_gen
                record["generated_tokens"] += n_gen
                record["turn_tokens"].append(n_gen)
                record["response_tokens"] = n_prompt - first_prompt_tokens + n_gen
                turn_log.write(json.dumps({"turn": record["num_chunks"], "prompt_tokens": n_prompt, "gen": n_gen,
                                           "finish": completion.finish_reason, "text": text}) + "\n")
                turn_log.flush()
                if record["response_tokens"] >= a.max_response or completion.finish_reason == "length":
                    record["end_reason"] = "budget"
                    break
                answer = text.rpartition(THINK_END)[2]
                blocks = JsonBlockProtocol.JSON_BLOCK_PATTERN.findall(answer)
                entry = {"round": record["num_chunks"], "state": result["state"], "prompt_tokens": n_prompt,
                         "gen_tokens": n_gen, "finish": completion.finish_reason, "gen_seconds": round(t_gen, 1),
                         "answer": answer.strip()[-2000:]}
                rounds.append(entry)
                try:
                    if not blocks:
                        raise InvalidAction("no ```json block found")
                    action = json.loads(blocks[-1])
                    JsonBlockProtocol.check_action(action, {p: tuple(q) for p, q in result["state"].items()})
                except (InvalidAction, json.JSONDecodeError) as exc:
                    record["end_reason"] = "parse"
                    record["parse_error"] = str(exc)
                    entry["parse_error"] = str(exc)
                    break
                t_sim = time.time()
                steps_before = len(history)
                result = await worker.call(env_args, history, action)
                history.extend(result["actions"])
                entry.update({"action": action, "outcome": result["outcome"], "executed": result["executed"],
                              "native_steps": [steps_before, len(history)], "reward": sum(result["rewards"]),
                              "done": result["done"], "state_after": result["state"],
                              "sim_seconds": round(time.time() - t_sim, 1)})
                self.save_frames(frames_dir, record["num_chunks"] + 1, result)
                record["num_chunks"] += 1
                record["executed_steps"] = len(history)
                record["done"] = result["done"]
                record["success"] = bool(result["done"]) and sum(result["rewards"]) > 0
                if result["done"]:
                    record["end_reason"] = "env"
                    break
                # the generation as it is (the eos token excluded by vLLM's text), then the next user turn
                content, images = self.user_content(result)
                full = self.render([messages[0], {"role": "user", "content": content}], True)
                assert full.startswith(system_render), "the system part of a render changed between turns"
                tail = full[len(system_render):]
                prompt = prompt + text + ("" if text.endswith("<|im_end|>") else "<|im_end|>") + "\n" + tail
                all_images.extend(images)
                if len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"]) + len(all_images) * 256 \
                        - first_prompt_tokens >= a.max_response:
                    record["end_reason"] = "budget"
                    break
        except Exception as exc:  # recorded: the episode counts as a failure
            record["end_reason"] = "error"
            record["error"] = repr(exc)[-2000:]
            if type(exc).__name__ == "EngineDeadError":
                self.engine_dead = True
        finally:
            turn_log.close()
            self.workers.put_nowait(worker)
        record["seconds"] = round(time.time() - start, 1)
        try:  # the recording never fails the episode
            (ep_dir / "trajectory.json").write_text(json.dumps({
                "env": env_args, "instruction": record.get("instruction"), "summary": record,
                "settings": {"effort": a.effort, "kv_cache_dtype": a.kv_cache_dtype, "temperature": a.temperature,
                             "top_p": a.top_p, "max_response": a.max_response, "model": a.model},
                "rounds": rounds,
                # LIBERO's 20 Hz native actions from the episode's initial state: a replay renders the full video
                "native_actions": history,
            }))
            labels = [f"{name} r0 (initial)"] + [
                f"r{r['round'] + 1} {r.get('outcome')} after {json.dumps(r.get('action'))}" for r in rounds
            ]
            self.write_gif(frames_dir, ep_dir / "rounds.gif", labels)
        except Exception as exc:
            print(f"recording of {name} failed: {exc!r}", flush=True)
        return record

    async def run(self) -> None:
        a = self.args
        done = set()
        if self.results_path.exists():
            for line in self.results_path.read_text().splitlines():
                r = json.loads(line)
                if r["end_reason"] != "error":
                    done.add(json.dumps(r["env"], sort_keys=True))
        todo = [e for e in episodes_of(a.episodes) if json.dumps(e, sort_keys=True) not in done]
        if a.limit:
            todo = todo[: a.limit]
        print(f"{len(done)} episodes done, {len(todo)} to run", flush=True)
        if not todo:
            return
        self.build_engine()
        for i in range(a.sessions):
            self.workers.put_nowait(Worker(a.libero_python, a.worker_script, a.mesa, a.libero_config,
                                           self.out / f"worker_{i}.log"))
        queue: asyncio.Queue = asyncio.Queue()
        for e in todo:
            queue.put_nowait(e)
        results_file = open(self.results_path, "a")

        async def runner():
            while not queue.empty() and not self.engine_dead:
                env_args = queue.get_nowait()
                record = await self.episode(env_args)
                if self.engine_dead:  # not recorded: the restart runs it again
                    return
                results_file.write(json.dumps(record) + "\n")
                results_file.flush()
                n = sum(1 for _ in open(self.results_path))
                rate = self.tokens_generated / max(time.time() - self.t0, 1)
                print(f"[{time.strftime('%H:%M:%S')}] {n} done | {env_args} success={record['success']} "
                      f"end={record['end_reason']} chunks={record['num_chunks']} resp_tok={record['response_tokens']} "
                      f"| gen {rate:.0f} tok/s", flush=True)

        stop = Path(a.out) / "STOP"
        tasks = [asyncio.create_task(runner()) for _ in range(a.sessions)]
        while not all(t.done() for t in tasks):
            if stop.exists():
                print("STOP file found: cancelling", flush=True)
                for t in tasks:
                    t.cancel()
                break
            await asyncio.sleep(10)
        await asyncio.gather(*tasks, return_exceptions=True)
        results_file.close()
        summarize(self.results_path, self.out / "summary.json")
        if self.engine_dead:
            print("ENGINE_DEAD: exiting for a restart", flush=True)
            os._exit(3)


def summarize(results_path: Path, summary_path: Path) -> None:
    # the last record of each episode, a non-error one before any error one
    last: dict = {}
    for line in results_path.read_text().splitlines():
        r = json.loads(line)
        key = json.dumps(r["env"], sort_keys=True)
        if key not in last or r["end_reason"] != "error" or last[key]["end_reason"] == "error":
            last[key] = r
    rows = list(last.values())
    by_suite: dict = {}
    for r in rows:
        s = by_suite.setdefault(r["env"]["suite"], {"n": 0, "success": 0, "end": {}})
        s["n"] += 1
        s["success"] += int(r["success"])
        s["end"][r["end_reason"]] = s["end"].get(r["end_reason"], 0) + 1
    for s in by_suite.values():
        s["rate"] = round(100 * s["success"] / s["n"], 1)
    total = {"n": len(rows), "success": sum(int(r["success"]) for r in rows)}
    total["rate"] = round(100 * total["success"] / max(total["n"], 1), 1)
    total["mean_suite_rate"] = round(sum(s["rate"] for s in by_suite.values()) / max(len(by_suite), 1), 1)
    summary = {"total": total, "by_suite": by_suite}
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--sessions", type=int, default=16)
    p.add_argument("--max-prompt", type=int, default=4096)
    p.add_argument("--max-response", type=int, default=256000)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--kv-cache-dtype", default="auto")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--effort", default="xhigh")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--engine-kwargs", default="{}")
    p.add_argument("--libero-python", default=f"{WORK}/envs/libero/bin/python")
    p.add_argument("--libero-config", default=f"{WORK}/envs/libero/etc/libero")
    p.add_argument("--mesa", default=f"{WORK}/envs/mesa")
    p.add_argument("--worker-script", default=str(Path(__file__).with_name("libero_worker.py")))
    p.add_argument("--summarize-only", action="store_true")
    a = p.parse_args()
    if a.summarize_only:
        summarize(Path(a.out) / "episodes.jsonl", Path(a.out) / "summary.json")
        return
    random.seed(a.seed)
    asyncio.run(Evaluator(a).run())


if __name__ == "__main__":
    main()
