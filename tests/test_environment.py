"""Smoke tests: is the LeRobot + Rerun environment correctly installed?

These need no robot and no network. If these fail, fix the install before anything else.
"""


def test_core_imports():
    import lerobot  # noqa: F401
    import rerun  # noqa: F401
    import torch  # noqa: F401


def test_feetech_sdk_present():
    # The Feetech motor SDK is what actually talks to the SO-101 servos.
    import scservo_sdk  # noqa: F401  (installed by lerobot[feetech])


def test_so101_classes_importable():
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    assert SO101Follower is not None
    assert SO101FollowerConfig is not None


def test_act_policy_available():
    """ACT is our from-scratch policy and must work out of the box."""
    from lerobot.policies.factory import get_policy_class

    assert get_policy_class("act") is not None


def test_smolvla_policy_available():
    """SmolVLA (the VLA path) needs the extra deps: uv pip install 'lerobot[smolvla]'.

    Skips (not fails) when transformers isn't installed -- finetuning may happen on
    Qualia/cloud instead of locally.
    """
    import importlib.util

    if importlib.util.find_spec("transformers") is None:
        import pytest

        pytest.skip("smolvla needs transformers -> uv pip install 'lerobot[smolvla]'")

    from lerobot.policies.factory import get_policy_class

    assert get_policy_class("smolvla") is not None
