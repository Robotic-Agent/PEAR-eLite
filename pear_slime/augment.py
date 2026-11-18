"""Conservative image augmentation for PEAR episode SFT.

LIBERO-plus / LIBERO-PRO are held out as the generalization test (perturbations: background, robot, camera,
language, noise, layout, light), so the augmentation stays small and leaves the text alone: per episode and per
camera, one transform drawn once and applied to every frame of that camera (the cameras are fixed during an
episode, so the frames stay consistent with each other and with the recorded actions):
  - a crop keeping 92-100% of each side, at a random offset, resized back (<= 4% shift of the view),
  - brightness, contrast and saturation factors in [0.9, 1.1].
No flips or rotations (they would contradict the actions' directions), no added noise, no colour/hue shifts, no
instruction rewrites. With probability 1 - p an episode keeps its original images.
Enabled by PEAR_SFT_AUG=<p> (e.g. 0.5) in pear_slime.sft_rollout.
"""

from __future__ import annotations

import random

from PIL import Image, ImageEnhance

CAMERAS_PER_TURN = 2  # agentview, wrist: the order of a user turn's images
MIN_KEEP = 0.92
JITTER = 0.1


def draw(rng: random.Random) -> dict:
    keep = rng.uniform(MIN_KEEP, 1.0)
    return {
        "keep": keep,
        "x": rng.uniform(0.0, 1.0 - keep),
        "y": rng.uniform(0.0, 1.0 - keep),
        "brightness": rng.uniform(1 - JITTER, 1 + JITTER),
        "contrast": rng.uniform(1 - JITTER, 1 + JITTER),
        "saturation": rng.uniform(1 - JITTER, 1 + JITTER),
    }


def apply(image: Image.Image, t: dict) -> Image.Image:
    w, h = image.size
    cw, ch = round(w * t["keep"]), round(h * t["keep"])
    left, top = round(w * t["x"]), round(h * t["y"])
    out = image.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BICUBIC)
    out = ImageEnhance.Brightness(out).enhance(t["brightness"])
    out = ImageEnhance.Contrast(out).enhance(t["contrast"])
    return ImageEnhance.Color(out).enhance(t["saturation"])


def augment_episode(images: list[Image.Image], p: float, rng: random.Random) -> list[Image.Image]:
    if p <= 0 or rng.random() >= p:
        return images
    transforms = [draw(rng) for _ in range(CAMERAS_PER_TURN)]
    return [apply(image, transforms[i % CAMERAS_PER_TURN]) for i, image in enumerate(images)]
