import asyncio
import os
import pickle
import sys

import pytest

from pear.agent_loop.episode import Episode
from pear.envs.games.minecraft.env import MinecraftEnv
from pear.sim.schemas import ServerContext

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(sys.platform != "linux", reason="SimulatorProcess is Linux-only"),
    pytest.mark.skipif(not os.environ.get("PEAR_MINECRAFT_RUNTIME"), reason="requires a running Minecraft arena"),
]

EPISODE = {"seed": 42, "max_ticks": 1}
NOOP = {"mouse": {"dx": 0, "dy": 0, "left": 0}}


def assert_initial_frame(result):
    assert not result["done"]
    assert result["state"] == {"episode": {"tick": 0, "ticks_left": 1}}
    [image] = result["images"]
    assert image.mode == "RGB" and image.width > 0 and image.height > 0
    assert image.getbbox() is not None


def assert_timeout(result):
    assert result["done"] and result["executed"] == 1
    assert result["outcome"] == "budget" and not Episode.is_success(result)


def test_native_minecraft_simulator_process(tmp_path):
    [process], _ = MinecraftEnv.setup_env(ServerContext({}, tmp_path, [], print))
    process.start()
    try:
        assert process.wait_ready(MinecraftEnv.START_TIMEOUT_S)
        assert_initial_frame(
            pickle.loads(process.call({"op": "open", "env_args": EPISODE}, MinecraftEnv.OPEN_TIMEOUT_S))
        )
        assert_timeout(pickle.loads(process.call({"op": "step", "action": NOOP}, MinecraftEnv.STEP_TIMEOUT_S)))
    finally:
        process.close()
        process.wait_or_kill(10)
    assert process.popen.returncode == 0


def test_native_minecraft_ray(tmp_path, local_ray_cluster):
    ray = pytest.importorskip("ray")
    from pear.sim.client import StatefulClient
    from pear.sim.server import ServerPool

    async def play():
        client = StatefulClient(MinecraftEnv)
        for uid in ("episode-1", "episode-2"):
            try:
                assert_initial_frame(await client.open(uid, EPISODE))
                assert_timeout(await client.step(uid, NOOP))
            finally:
                await client.close(uid)

    # Ray rewrites 127.0.0.1 to the default routed address, which may be a VPN/WSL interface.
    with local_ray_cluster(address="local", _node_ip_address="127.0.0.2", num_cpus=1, resources={"minecraft": 1}):
        pool = ServerPool.start(MinecraftEnv, {"server_n": 1, "log_dir": str(tmp_path)})
        try:
            asyncio.run(asyncio.wait_for(play(), timeout=300))
        finally:
            ray.kill(pool)
