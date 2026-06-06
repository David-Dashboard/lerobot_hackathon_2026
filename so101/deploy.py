"""Run a policy on an arm -- the deployment loop.

A `Policy` maps current joints -> target joints. We ship stub policies so the
deploy path is demoable with NO trained model and NO hardware (e.g. on the mock
arm, `SinePolicy` makes it move autonomously). A real LeRobot policy plugs in via
`LeRobotPolicy` without changing the loop.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Protocol

from .interface import RobotArm
from .obs import SO101_JOINTS


class Policy(Protocol):
    def select_action(self, joints: dict[str, float]) -> dict[str, float]: ...


class HoldPolicy:
    """Command the current position (no motion)."""

    def select_action(self, joints: dict[str, float]) -> dict[str, float]:
        return dict(joints)


class SinePolicy:
    """Autonomous sine sweep -- a visible 'it's running' demo on the mock arm."""

    def __init__(self, amplitude: float = 40.0, period_s: float = 4.0):
        self.amplitude = amplitude
        self.period_s = period_s
        self._t0: float | None = None

    def select_action(self, joints: dict[str, float]) -> dict[str, float]:
        if self._t0 is None:
            self._t0 = time.time()
        phase = 2 * math.pi * (time.time() - self._t0) / self.period_s
        return {j: self.amplitude * math.sin(phase + i) for i, j in enumerate(SO101_JOINTS)}


def build_observation(joints: dict[str, float], images: dict[str, "object"], device: str = "cpu",
                      task: str | None = None) -> dict:
    """Build a batched observation dict a LeRobot policy's ``select_action`` expects.

    Inverse of the dataset write path in ``record.py``: plain joint dict + uint8
    ``HxWxC`` images -> batched tensors on ``device``:

    * ``observation.state`` -> ``(1, J)`` float32
    * ``observation.images.<name>`` -> ``(1, 3, H, W)`` float32 in ``[0, 1]`` (CHW)
    * ``task`` -> ``[task]`` (for task-conditioned / VLA policies)

    Torch is imported lazily so the package stays import-light without it.
    """
    import torch

    state = torch.tensor([[joints[j] for j in SO101_JOINTS]], dtype=torch.float32, device=device)
    obs: dict = {"observation.state": state}
    for name, img in (images or {}).items():
        import numpy as np

        arr = np.asarray(img).astype(np.float32) / 255.0  # HxWxC [0,1]
        chw = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W)
        key = name if name.startswith("observation.images.") else f"observation.images.{name}"
        obs[key] = chw
    if task is not None:
        obs["task"] = [task]
    return obs


class LeRobotPolicy:
    """Adapter around a trained LeRobot policy loaded from a checkpoint dir / HF repo.

    Lazy-imports LeRobot + torch. A vision policy needs camera frames, so this is
    driven by ``eval_offline.py`` (and a real arm + cameras) rather than the
    stub-only closed loop in ``run_policy`` -- hence ``select_action`` here takes
    an extra ``images`` arg. Not exercised in the hardware-free tests (needs a
    real checkpoint).
    """

    def __init__(self, path: str, device: str = "cpu", task: str | None = None):
        self.path = path
        self.device = device
        self.task = task
        self._policy = None

    def _ensure(self):
        if self._policy is not None:
            return
        # Pick the concrete policy class from the checkpoint config's `type`, then
        # load weights with from_pretrained (the robust path across lerobot 0.4.x).
        from lerobot.policies.factory import get_policy_class

        cfg_type = _read_checkpoint_type(self.path)
        policy_cls = get_policy_class(cfg_type)
        policy = policy_cls.from_pretrained(self.path)
        policy.to(self.device)
        policy.eval()
        if hasattr(policy, "reset"):
            policy.reset()
        self._policy = policy

    def reset(self):
        """Clear the policy's internal action queue (call at episode start)."""
        self._ensure()
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def select_action(self, joints: dict[str, float],
                      images: dict[str, "object"] | None = None) -> dict[str, float]:
        import torch

        self._ensure()
        obs = build_observation(joints, images or {}, device=self.device, task=self.task)
        with torch.no_grad():
            action = self._policy.select_action(obs)
        values = action.squeeze(0).detach().cpu().numpy().tolist()
        return {j: float(v) for j, v in zip(SO101_JOINTS, values)}


def _read_checkpoint_type(path: str) -> str:
    """Read the policy ``type`` from a checkpoint's config.json (local or HF repo)."""
    import json
    from pathlib import Path

    local = Path(path) / "config.json"
    if local.exists():
        with open(local) as fh:
            return json.load(fh)["type"]
    # Hub repo: fetch just the config file.
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(repo_id=path, filename="config.json")) as fh:
        return json.load(fh)["type"]


def load_policy(name_or_path: str) -> Policy:
    """'sine'/'hold' -> stub policy; anything else -> a trained LeRobot policy path."""
    key = (name_or_path or "").strip().lower()
    if key in ("sine", "demo", ""):
        return SinePolicy()
    if key in ("hold", "zero"):
        return HoldPolicy()
    return LeRobotPolicy(name_or_path)


def run_policy(
    arm: RobotArm,
    policy: Policy,
    *,
    steps: int = 200,
    fps: int = 30,
    should_stop: Callable[[], bool] | None = None,
    on_step: Callable[[int, int], None] | None = None,
) -> dict:
    """Closed loop: read -> policy -> command, for `steps` (or until should_stop)."""
    if not arm.is_connected:
        arm.connect()
    period = 1.0 / fps
    done = 0
    for t in range(steps):
        if should_stop and should_stop():
            break
        action = policy.select_action(arm.read_joints())
        arm.write_joints(action)
        done = t + 1
        if on_step:
            on_step(done, steps)
        time.sleep(period)
    return {"steps": done}
