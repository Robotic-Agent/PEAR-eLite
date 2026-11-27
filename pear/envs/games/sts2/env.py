from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

from pear.envs.games.common import GameEnv, game_result
from pear.sim.schemas import EnvResult

_PHASES = (
    "combat",
    "card_reward",
    "map",
    "rest",
    "shop",
    "loot",
    "complete",
    "event",
    "ancient",
    "deck_selection",
    "treasure",
)
_INTENTS = ("attack", "defend", "buff", "debuff", "unknown")


@cache
def _card_names() -> dict[int, str]:
    from sts2_gym.commands import card_name_by_id

    return card_name_by_id()


def _make_run(seed: int, max_steps: int):
    from sts2_gym import Sts2RunEnv

    return Sts2RunEnv(seed=seed, max_episode_steps=max_steps)


def _indexed(values):
    return {str(i): int(value) for i, value in enumerate(values) if value}


class SlayTheSpire2Env(GameEnv):
    """Text-only PEAR adapter for the run-level NativeAOT Gym environment used by sts2-rl."""

    PYTHON_VARIABLE = "PEAR_STS2_PYTHON"
    SCHEMA = {"run": ("action", "target")}
    CAMERAS = ()
    ENVIRONMENT_PROMPT = Path(__file__).with_name("environment_prompt.md").read_text(encoding="utf-8").strip()
    INSTRUCTION = "Win the Ironclad run as defined by the headless emulator."

    def __init__(self, seed: int, max_steps: int = 1000) -> None:
        if type(seed) is not int or seed < 0 or type(max_steps) is not int or max_steps < 1:
            raise ValueError("expected a non-negative integer seed and a positive integer max_steps")
        self.seed, self.max_steps = seed, max_steps
        self.run = None
        self.legal: set[int] = set()
        self.done = False

    @classmethod
    def task_name(cls, env_args: dict[str, Any]) -> str:
        return "sts2/ironclad-run"

    def open(self) -> EnvResult:
        if self.run is not None:
            raise RuntimeError("run already open")
        self.run = _make_run(self.seed, self.max_steps)
        try:
            obs, info = self.run.reset()
            self.done = False
            return self._observe(obs, info)
        except BaseException:
            self.close()
            raise

    def _observe(self, obs, info, action=None) -> EnvResult:
        self.legal = {i for i, allowed in enumerate(self.run.action_masks()) if allowed}
        phase = int(info["phase"])
        state = {
            _PHASES[phase]: {"floor": info["floor"], "deck_size": info["deck_size"], "gold": info["gold"]},
            "player": {"hp": info["player_hp"], "max_hp": info["player_max_hp"]},
            "available_actions": {str(i): 1 for i in sorted(self.legal)},
        }
        if phase == 0:
            state["player"].update(block=int(obs[2]), energy=int(obs[3]))
            state["pile_counts"] = dict(zip(("draw", "discard", "exhaust"), map(int, obs[5:8])))
            hand = [(int(obs[i]), int(obs[i + 1])) for i in range(8, 28, 2) if obs[i]]
            for i, (card, upgraded) in enumerate(hand):
                name = _card_names().get(card, "unknown")
                state[f"hand_{i}_{name}"] = {"card_id": card, "upgraded": upgraded}
            state["end_turn"] = {"action": len(hand)}
            state["potions"] = {str(len(hand) + 1 + int(slot)): item for slot, item in _indexed(obs[28:34:2]).items()}
            for i, offset in enumerate(range(54, 144, 15)):
                hp, max_hp, block, intent, magnitude = map(int, obs[offset : offset + 5])
                if hp > 0:
                    state[f"enemy_{i}"] = {"hp": hp, "max_hp": max_hp, "block": block, _INTENTS[intent]: magnitude}
        elif phase == 1:
            for slot, card in _indexed(info["card_rewards"]).items():
                state[f"reward_{slot}_{_card_names().get(card, 'unknown')}"] = {"card_id": card}
        elif phase == 2:
            # The observation keeps the original slot indices. Do not show seeded encounter previews.
            state["map_node_types"] = _indexed(obs[176:180])
        elif phase == 4:
            state["shop_costs"] = {str(i): info["shop_costs"][i] for i in sorted(self.legal) if i < 14}
            for slot, card in _indexed(info["shop_cards"]).items():
                state[f"shop_{slot}_{_card_names().get(card, 'unknown')}"] = {"card_id": card}
            state["shop_relics"] = _indexed(info["shop_relics"])
            state["shop_potions"] = _indexed(info["shop_potions"])
        elif phase in (5, 10):
            state["loot_ids"] = _indexed(info["pending_rewards"])
        elif phase == 7:
            state["event"]["id"] = info["event_id"]
        elif phase == 8:
            state["ancient_options"] = _indexed(info["neow_options"])
        return game_result(self.INSTRUCTION, state, [], native_action=action, outcome="stepped" if action else None)

    def step(self, action: Any) -> EnvResult:
        if self.run is None or self.done:
            raise RuntimeError("no active run")
        index, target = action["run"]["action"], action["run"]["target"]
        if index not in self.legal or not (-1 <= target <= 31 and int(target) == target):
            self.done = True
            return game_result(self.INSTRUCTION, None, [], done=True, outcome="invalid_action")
        index, target = int(index), int(target)
        obs, reward, terminated, truncated, info = self.run.step(index, target)
        # Native invalid target/selection (the action mask itself is one-dimensional).
        if reward == -1.0 and not (terminated or truncated):
            self.done = True
            return game_result(self.INSTRUCTION, None, [], done=True, outcome="invalid_action")
        self.done = bool(terminated or truncated)
        if self.done:
            won = bool(terminated and info["player_won"])
            return game_result(
                self.INSTRUCTION,
                None,
                [],
                native_action=[index, target],
                done=True,
                won=won,
                outcome="won" if won else "budget" if truncated else "lost",
            )
        return self._observe(obs, info, [index, target])

    def close(self) -> None:
        if self.run is not None:
            run, self.run = self.run, None
            run.close()
        self.done = True
