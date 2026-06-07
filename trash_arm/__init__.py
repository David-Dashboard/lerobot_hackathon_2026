"""Trash-collecting arm -- Phase 1 modular pipeline.

perceive -> localize -> decide -> motion -> loop. Each stage lives in its own
module behind a small interface so it can be unit-tested and swapped (e.g. the
scripted grasp -> a learned ACT skill) without touching the others.

Only `perception` (transformers) and `motion`/`camera` (LeRobot/cv2 + hardware)
pull heavy deps, and they do so lazily, so importing the pure stages
(`localize`, `decide`, `safety`, `config`) stays cheap and hardware-free.
"""

from .config import load_config

__all__ = ["load_config"]
