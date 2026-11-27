import os
import pickle
import sys

import pytest

from pear.agent_loop.episode import Episode
from pear.envs.games.sts2.env import SlayTheSpire2Env
from pear.sim.schemas import ServerContext

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("PEAR_TEST_NATIVE_STS2") != "1", reason="requires a built sts2_gym runtime"),
]


def test_native_sts2_run():
    def run():
        env = SlayTheSpire2Env(seed=42, max_steps=24)
        episode = Episode({"seed": 42})
        observations = []
        try:
            result = env.open()
            assert result["images"] == []
            while not result["done"]:
                state = result["state"]
                observations.append(state)
                index = int(next(iter(state["available_actions"])))
                result = env.step({"run": {"action": index, "target": -1}})
                episode.update(result)
            assert result["outcome"] == "budget"
            assert any("combat" in state for state in observations)
            assert any("map" in state for state in observations)
            assert len(episode.actions) == 24 and not episode.success
            return observations, episode.actions
        finally:
            env.close()

    assert run() == run()


@pytest.mark.skipif(sys.platform != "linux", reason="SimulatorProcess is Linux-only")
def test_native_sts2_simulator_process(tmp_path, monkeypatch):
    monkeypatch.setenv("PEAR_STS2_PYTHON", sys.executable)
    [process], _ = SlayTheSpire2Env.setup_env(ServerContext({}, tmp_path, [], print))
    process.start()
    try:
        assert process.wait_ready(30)
        result = pickle.loads(process.call({"op": "open", "env_args": {"seed": 42, "max_steps": 1}}, 30))
        assert result["images"] == [] and not result["done"]
        index = int(next(iter(result["state"]["available_actions"])))
        result = pickle.loads(process.call({"op": "step", "action": {"run": {"action": index, "target": -1}}}, 30))
        assert result["outcome"] == "budget" and not Episode.is_success(result)
    finally:
        process.close()
        process.wait_or_kill(10)
    assert process.popen.returncode == 0
