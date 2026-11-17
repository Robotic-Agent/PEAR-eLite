"""A LIBERO simulator worker, run in PEAR's LIBERO environment's Python: one JSON request per stdin line
({"env_args", "history", "action"}), one JSON reply per stdout line (LiberoEnv.rollout's result, frames as PNG base64).
Usage: LIBERO_MESA_DIR=... python libero_worker.py (PYTHONPATH holds the PEAR repository)."""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import traceback


def main() -> None:
    out = sys.stdout
    sys.stdout = sys.stderr  # LIBERO and robosuite print: keep the protocol channel clean
    from pear.envs.libero.env import LiberoEnv

    envs: dict[str, LiberoEnv] = {}
    out.write(json.dumps({"ready": True}) + "\n")
    out.flush()
    for line in sys.stdin:
        try:
            req = json.loads(line)
            key = json.dumps(req["env_args"], sort_keys=True)
            if key not in envs:
                envs.clear()  # one episode at a time per worker
                envs[key] = LiberoEnv(**req["env_args"])
            res = envs[key].rollout(req["history"], req["action"])
            if res["images"] is not None:
                frames = []
                for image in res["images"]:
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    frames.append(base64.b64encode(buf.getvalue()).decode())
                res["images"] = frames
            res["actions"] = [[float(x) for x in a] for a in res["actions"]]
            res["rewards"] = [float(r) for r in res["rewards"]]
            res["done"] = bool(res["done"])
            reply = {"result": res}
        except Exception:
            reply = {"error": traceback.format_exc()}
        out.write(json.dumps(reply) + "\n")
        out.flush()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
