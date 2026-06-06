"""SO-101 hackathon package: a decoupled arm interface, a mock, a real adapter,
and a remote-control server/client.

Light imports only -- `server` (fastapi) and `client` (httpx) are imported from
their submodules on demand, and `real`/LeRobot stays lazy. So `import so101` never
pulls torch.
"""

from .factory import make_arm
from .interface import JointPositions, RobotArm
from .mock import MockArm
from .obs import SO101_JOINTS, format_joint_line, joints_to_action, observation_to_joints

__all__ = [
    "RobotArm",
    "JointPositions",
    "MockArm",
    "make_arm",
    "SO101_JOINTS",
    "observation_to_joints",
    "joints_to_action",
    "format_joint_line",
]
