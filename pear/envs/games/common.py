from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pear.sim.env import StatefulEnv
from pear.sim.schemas import EnvResult, ServerContext

if TYPE_CHECKING:
    from pear.sim.process import SimulatorProcess


class GameEnv(StatefulEnv):
    SYSTEM_PROMPT_TEMPLATE = Path(__file__).with_name("system_prompt.md").read_text(encoding="utf-8")
    OBSERVATION_REMINDER = "End your response with one json block containing the next action."
    SERVER_RESOURCES = {"num_cpus": 1}
    START_TIMEOUT_S = 120.0
    OPEN_TIMEOUT_S = 120.0
    STEP_TIMEOUT_S = 90.0
    PYTHON_VARIABLE: str

    @classmethod
    def setup_env(cls, context: ServerContext) -> tuple[list[SimulatorProcess], int]:
        from pear.sim.process import SimulatorProcess

        process = SimulatorProcess(
            "sim",
            cls,
            os.environ.get(cls.PYTHON_VARIABLE) or sys.executable,
            [],
            dict(os.environ),
            SimulatorProcess.PEAR_ROOT,
            context.log_dir,
        )
        return [process], 1


def game_result(
    instruction: str,
    state: dict[str, dict[str, float]] | None,
    images: list[Any],
    *,
    native_action: list[int | float | bool] | None = None,
    done: bool = False,
    won: bool = False,
    outcome: str | None = None,
) -> EnvResult:
    # The PEAR agent loop reads positive terminal reward as success. Neither dense shaping nor timeout can count.
    return {
        "images": None if done else images,
        "state": state,
        "actions": [] if native_action is None else [native_action],
        "rewards": [] if native_action is None else [float(won)],
        "done": done,
        "executed": int(native_action is not None),
        "outcome": outcome,
        "instruction": instruction,
    }
