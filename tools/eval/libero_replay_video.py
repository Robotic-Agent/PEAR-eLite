"""Renders the full 20 Hz video of recorded episodes by replaying their native actions in LIBERO (PEAR's LiberoEnv,
the same initial state, seed and stabilization as the eval). Runs in the LIBERO environment's Python with the Mesa
variables of the eval's workers (replay_videos.sh sets them).
Usage: python libero_replay_video.py <episodes dir> [--only <name>,...] [--fps 20] [--check]
Writes <episode>/replay.mp4 (agentview | wrist, frames rotated as the model sees them) and checks that the replay ends
where the recording says (the last state, and the success)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def replay(ep_dir: Path, fps: int) -> dict:
    import imageio.v2 as imageio
    from pear.envs.libero.controller import MotionController
    from pear.envs.libero.env import LiberoEnv

    traj = json.loads((ep_dir / "trajectory.json").read_text())
    env = LiberoEnv(**traj["env"])
    observation = env._open_scene()
    writer = imageio.get_writer(ep_dir / "replay.mp4", fps=fps, codec="libx264", quality=7, macro_block_size=1)

    def frame(obs) -> np.ndarray:
        views = [obs[env.OBSERVATION_KEY_BY_CAMERA[c]][::-1, ::-1] for c in env.CAMERAS]
        return np.ascontiguousarray(np.concatenate(views, axis=1), dtype=np.uint8)

    writer.append_data(frame(observation))
    rewards = []
    for native_action in traj["native_actions"]:
        observation, reward, _, _ = env.scene.step(np.asarray(native_action, dtype=np.float64))
        rewards.append(float(reward))
        writer.append_data(frame(observation))
    writer.close()
    state = MotionController.state(env.scene, observation)
    env._close_scene()
    rounds = [r for r in traj["rounds"] if "state_after" in r]
    expected = rounds[-1]["state_after"] if rounds else None
    return {
        "episode": ep_dir.name,
        "steps": len(traj["native_actions"]),
        "replay_success": sum(rewards) > 0,
        "recorded_success": traj["summary"]["success"],
        # the recording's last state is None once the episode ended (its frames are not rendered)
        "state_match": expected is None or expected == state,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("episodes_dir")
    p.add_argument("--only", default="")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--skip-done", action="store_true")
    a = p.parse_args()
    only = set(filter(None, a.only.split(",")))
    for ep_dir in sorted(Path(a.episodes_dir).iterdir()):
        if not (ep_dir / "trajectory.json").exists() or (only and ep_dir.name not in only):
            continue
        if a.skip_done and (ep_dir / "replay.mp4").exists():
            continue
        try:
            print(json.dumps(replay(ep_dir, a.fps)), flush=True)
        except Exception as exc:
            print(json.dumps({"episode": ep_dir.name, "error": repr(exc)}), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
