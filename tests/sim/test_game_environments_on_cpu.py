import json
import pkgutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from pear.agent_loop.episode import Episode
from pear.agent_loop.protocol import JsonBlockProtocol
from pear.envs.games.minecraft import env as minecraft
from pear.envs.games.sts2 import env as sts2
from pear.sim.env import EpisodeLost, StatefulEnv
from pear.sim.schemas import ServerContext

GAMES = Path(sts2.__file__).parents[1]


@pytest.mark.parametrize("game", ["sts2", "minecraft"])
def test_episode_config_resolves_without_game_engine(game, tmp_path):
    options = json.loads((GAMES / game / "configs/smoke.json").read_text())
    assert options["data_source"] in options["env"]
    env_class = pkgutil.resolve_name(options["env_class"])
    assert issubclass(env_class, StatefulEnv)
    args = {key: values[0] for key, values in options["env"].items()}
    env = env_class(**args)
    prompt = JsonBlockProtocol.system_prompt(env.SYSTEM_PROMPT_TEMPLATE, env.ENVIRONMENT_PROMPT, env.INSTRUCTION)
    assert "playing a game" in prompt and "{instruction}" not in prompt
    assert env.task_name(args)
    [process], starts = env_class.setup_env(ServerContext({}, tmp_path, [], print))
    assert starts == 1 and options["env_class"] in process.command


class FakeRun:
    def __init__(self, transition):
        self.transition = transition
        self.actions = []
        self.closed = False

    def reset(self):
        return [0] * 199, {
            "phase": 8,
            "floor": 1,
            "gold": 99,
            "deck_size": 11,
            "player_hp": 64,
            "player_max_hp": 80,
            "neow_options": (10, 0, 13),
        }

    def action_masks(self):
        return [True, False, True]

    def step(self, index, target):
        self.actions.append((index, target))
        obs, info = self.reset()
        reward, terminated, truncated, player_won = self.transition
        return obs, reward, terminated, truncated, {**info, "player_won": player_won}

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    ("transition", "done", "success", "reward", "outcome"),
    [
        ((0.5, False, False, False), False, False, [0.0], "stepped"),
        ((0.5, False, True, False), True, False, [0.0], "budget"),
        ((0.5, True, False, False), True, False, [0.0], "lost"),
        ((0.5, True, False, True), True, True, [1.0], "won"),
        ((-1.0, False, False, False), True, False, [], "invalid_action"),
    ],
)
def test_sts2_rewards_and_receipts(monkeypatch, transition, done, success, reward, outcome):
    run = FakeRun(transition)
    monkeypatch.setattr(sts2, "_make_run", lambda *_: run)
    env = sts2.SlayTheSpire2Env(seed=42)
    first = env.open()
    assert first["images"] == [] and first["executed"] == 0
    assert first["state"]["available_actions"] == {"0": 1, "2": 1}
    action = {"run": {"action": 2, "target": -1}}
    JsonBlockProtocol.check_action(action, env.SCHEMA)
    result = env.step(action)
    assert run.actions == [(2, -1)]
    assert result["done"] == done and result["rewards"] == reward and result["outcome"] == outcome
    assert result["actions"] == ([[2, -1]] if reward else [])
    assert Episode.is_success(result) == success
    env.close()
    assert run.closed


def test_sts2_masked_action_does_not_enter_runtime(monkeypatch):
    run = FakeRun(None)
    monkeypatch.setattr(sts2, "_make_run", lambda *_: run)
    env = sts2.SlayTheSpire2Env(seed=42)
    env.open()
    assert env.step({"run": {"action": 1, "target": -1}})["outcome"] == "invalid_action"
    assert run.actions == []
    env.close()


def test_minecraft_measures_ticks_and_exposes_only_rgb(monkeypatch):
    pytest.importorskip("PIL")
    frame = SimpleNamespace(tick=0, width=2, height=1, rgb=b"\x01\x02\x03" * 2, target_position=[1, 2, 3])
    arena = SimpleNamespace(reset=lambda *_: frame, close=lambda: None)
    calls = []

    def tick(current, dx, dy, left):
        assert current is arena
        calls.append((dx, dy, left))
        frame.tick += 1
        return SimpleNamespace(observation=frame, reward=1, terminated=True, truncated=False)

    monkeypatch.setenv("PEAR_MINECRAFT_RUNTIME", "not-accessed-by-test")
    monkeypatch.setattr(minecraft, "_connect_arena", lambda *_: arena)
    monkeypatch.setattr(minecraft, "_tick", tick)
    env = minecraft.MinecraftEnv(seed=42, max_ticks=4)
    result = env.open()
    assert result["images"][0].tobytes() == frame.rgb
    assert result["state"] == {"episode": {"tick": 0, "ticks_left": 4}}
    action = {"mouse": {"dx": 1.5, "dy": -2, "left": 1}}
    JsonBlockProtocol.check_action(action, env.SCHEMA)
    result = env.step(action)
    assert calls == [(1.5, -2.0, True)]
    assert result["images"] is None and result["actions"] == [[1.5, -2.0, True]]
    assert result["outcome"] == "hit" and Episode.is_success(result)
    with pytest.raises(RuntimeError, match="no active"):
        env.step(action)
    env.close()
    assert len(calls) == 1


def test_minecraft_transport_loss_is_not_retried(monkeypatch):
    env = minecraft.MinecraftEnv(seed=42)
    env.arena = object()
    calls = []

    def lose(*args):
        calls.append(args)
        raise EpisodeLost

    monkeypatch.setattr(minecraft, "_tick", lose)
    action = {"mouse": {"dx": 0, "dy": 0, "left": 0}}
    with pytest.raises(EpisodeLost):
        env.step(action)
    with pytest.raises(RuntimeError, match="no active"):
        env.step(action)
    assert len(calls) == 1
