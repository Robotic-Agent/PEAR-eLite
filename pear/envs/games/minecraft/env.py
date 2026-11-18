from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from pear.envs.games.common import GameEnv, game_result
from pear.sim.env import EpisodeLost
from pear.sim.schemas import EnvResult

logger = logging.getLogger(__name__)


def _connect_arena(output: Path, max_ticks: int):
    from minecraft_gym.games.minecraft.coordinator import CoordinatedEnvironment

    manifest = Path(os.environ["PEAR_MINECRAFT_RUNTIME"])
    runtime = json.loads(manifest.read_text(encoding="utf-8"))
    return CoordinatedEnvironment(
        Path(runtime["coordinator"]),
        Path(runtime["server_endpoint"]),
        Path(runtime["client_endpoint"]),
        runtime["player_uuid"],
        output,
        max_ticks=max_ticks,
        timeout=45,
    )


def _tick(arena, dx: float, dy: float, left: bool):
    from minecraft_gym.core.actions import TickAction
    from minecraft_gym.games.minecraft.rust_runtime import ProtocolError

    try:
        return arena.step(TickAction(dx, dy, left))
    except (OSError, ProtocolError) as exc:
        raise EpisodeLost("Minecraft Step receipt failed; the action is not retried") from exc


class MinecraftEnv(GameEnv):
    """Horizon RGB + Pumpkin's server-authoritative moving sheep task, through the Rust Step coordinator."""

    PYTHON_VARIABLE = "PEAR_MINECRAFT_PYTHON"
    SERVER_RESOURCES = {"num_cpus": 1, "resources": {"minecraft": 1}}
    SCHEMA = {"mouse": ("dx", "dy", "left")}
    CAMERAS = ("first_person",)
    ENVIRONMENT_PROMPT = Path(__file__).with_name("environment_prompt.md").read_text(encoding="utf-8").strip()
    INSTRUCTION = "Aim at the moving red sheep and hit it with a left click."
    DIFFICULTIES = ("easy", "medium", "hard")

    def __init__(self, seed: int, difficulty: str = "easy", max_ticks: int = 160) -> None:
        if type(seed) is not int or not -(1 << 63) <= seed < (1 << 63):
            raise ValueError("expected a signed 64-bit integer seed")
        if difficulty not in self.DIFFICULTIES or type(max_ticks) is not int or max_ticks < 1:
            raise ValueError("expected easy/medium/hard and a positive integer max_ticks")
        self.seed, self.difficulty, self.max_ticks = seed, difficulty, max_ticks
        self.arena = None
        self.work: TemporaryDirectory | None = None
        self.done = False

    @classmethod
    def task_name(cls, env_args: dict[str, Any]) -> str:
        return f"minecraft/moving-sheep/{env_args.get('difficulty', 'easy')}"

    @classmethod
    def server_n(cls, config: dict[str, Any], cluster_resources: dict[str, float]) -> int:
        if int(config["server_n"]) != 1 or cluster_resources.get("minecraft") != 1:
            raise ValueError('Minecraft requires exactly one prestarted arena and one Ray resource {"minecraft": 1}')
        return 1

    def open(self) -> EnvResult:
        if self.arena is not None:
            raise RuntimeError("arena already open")
        if not os.environ.get("PEAR_MINECRAFT_RUNTIME"):
            raise RuntimeError("set PEAR_MINECRAFT_RUNTIME to the arena manifest from minecraft.launch")
        self.work = TemporaryDirectory(prefix="pear-minecraft-")
        try:
            self.arena = _connect_arena(Path(self.work.name) / "coordinator", self.max_ticks)
            self.done = False
            return self._observe(self.arena.reset(self.seed, self.difficulty))
        except BaseException:
            self.close()
            raise

    def _observe(self, frame, action=None) -> EnvResult:
        from PIL import Image

        state = {"episode": {"tick": frame.tick, "ticks_left": self.max_ticks - frame.tick}}
        images = [Image.frombytes("RGB", (frame.width, frame.height), frame.rgb)]
        return game_result(self.INSTRUCTION, state, images, native_action=action, outcome="stepped" if action else None)

    def step(self, action: Any) -> EnvResult:
        if self.arena is None or self.done:
            raise RuntimeError("no active Minecraft episode")
        dx, dy, left = (action["mouse"][key] for key in self.SCHEMA["mouse"])
        if not (abs(dx) <= 10000 and abs(dy) <= 10000 and left in (0, 1)):
            self.done = True
            return game_result(self.INSTRUCTION, None, [], done=True, outcome="invalid_action")
        dx, dy, left = float(dx), float(dy), bool(left)
        applied = [dx, dy, left]
        try:
            transition = _tick(self.arena, dx, dy, left)
        except EpisodeLost:
            self.done = True
            raise
        self.done = transition.terminated or transition.truncated
        if self.done:
            won = transition.terminated and transition.reward > 0
            return game_result(
                self.INSTRUCTION,
                None,
                [],
                native_action=applied,
                done=True,
                won=won,
                outcome="hit" if won else "budget" if transition.truncated else "lost",
            )
        return self._observe(transition.observation, applied)

    def close(self) -> None:
        self.done = True
        if self.arena is not None:
            arena, self.arena = self.arena, None
            try:
                arena.close()
            except Exception:
                logger.exception("Minecraft coordinator ended with an error during close")
        if self.work is not None:
            work, self.work = self.work, None
            work.cleanup()
