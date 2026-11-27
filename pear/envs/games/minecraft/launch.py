"""Start one pinned Minecraft-GYM-rs arena, reusing its validated offscreen launcher."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
from pathlib import Path
from threading import Event


def _load_launcher(root: Path):
    spec = importlib.util.spec_from_file_location(
        "minecraft_gym_native_launcher", root / "scripts/smoke_horizon_pumpkin.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _smoke(manifest: Path):
    from minecraft_gym.core.actions import parse_action
    from minecraft_gym.core.interfaces import Observation
    from minecraft_gym.games.minecraft.policies import MinecraftColorPolicy

    from pear.agent_loop.episode import Episode
    from pear.envs.games.minecraft.env import MinecraftEnv

    os.environ["PEAR_MINECRAFT_RUNTIME"] = str(manifest)
    env = MinecraftEnv(seed=42, max_ticks=80)
    policy = MinecraftColorPolicy()
    policy.reset(42)
    try:
        result = env.open()
        size = result["images"][0].size
        for tick in range(80):
            image = result["images"][0]
            block = parse_action(policy.act(Observation(tick, *image.size, image.tobytes()))).block
            result = env.step({"mouse": {"dx": block.dx, "dy": block.dy, "left": 1}})
            if result["done"]:
                break
            if result["state"]["episode"]["tick"] != tick + 1:
                raise AssertionError("unexpected Minecraft tick")
        if not Episode.is_success(result):
            raise AssertionError("RGB policy did not reach a server-confirmed hit through the PEAR adapter")
        proof = {"hit": True, "ticks": tick + 1, "rgb_size": list(size)}
    finally:
        env.close()

    short_episode = MinecraftEnv(seed=42, max_ticks=1)
    try:
        short_episode.open()
        result = short_episode.step({"mouse": {"dx": 0, "dy": 0, "left": 0}})
        if not result["done"] or result["outcome"] != "budget" or Episode.is_success(result):
            raise AssertionError("one-tick budget did not produce a clean failure")
    finally:
        short_episode.close()
    return {**proof, "second_episode_timeout": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamebench-root", required=True, type=Path)
    parser.add_argument("--coordinator", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true", help="run two PEAR adapter episodes, then shut down")
    args, upstream_args = parser.parse_known_args()
    launcher = _load_launcher(args.gamebench_root.resolve())
    runtime_args = launcher.parser().parse_args(upstream_args)
    coordinator = args.coordinator.resolve(strict=True)
    stop = Event()

    def request_stop(_signum, _frame):
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def serve(arena, output, server_endpoint, client_endpoint):
        player_uuid = arena.server.list_environments()["environments"][0]["entity_uuid"]
        arena.close()  # the Rust coordinator takes the exclusive lease per episode
        arena.client.close()
        arena.server.close()
        manifest = output / "pear-arena.json"
        pending = manifest.with_suffix(".tmp")
        try:
            pending.write_text(
                json.dumps(
                    {
                        "coordinator": str(coordinator),
                        "server_endpoint": str(server_endpoint),
                        "client_endpoint": str(client_endpoint),
                        "player_uuid": player_uuid,
                    }
                ),
                encoding="utf-8",
            )
            pending.replace(manifest)
            print(f"PEAR_MINECRAFT_RUNTIME={manifest}", flush=True)
            if args.smoke:
                return {"pear_smoke": _smoke(manifest)}
            stop.wait()
            return {}
        finally:
            manifest.unlink(missing_ok=True)
            pending.unlink(missing_ok=True)
            # Return shutdown control to the upstream launcher, which quits and joins both child processes.
            arena.client = launcher.HorizonControl(client_endpoint, timeout=10)
            arena.server = launcher.PumpkinControl(server_endpoint, timeout=10)

    launcher.run(runtime_args, experiment=serve)


if __name__ == "__main__":
    main()
