"""Arm role-detection decision logic -- pure, no hardware."""

from so101.arm_id import decide_roles, movement_score


def test_movement_score_is_summed_peak_to_peak():
    samples = [
        {"shoulder_pan": 0.0, "elbow_flex": 10.0},
        {"shoulder_pan": 5.0, "elbow_flex": 10.0},
        {"shoulder_pan": -3.0, "elbow_flex": 12.0},
    ]
    # pan range = 5 - (-3) = 8 ; elbow range = 12 - 10 = 2
    assert movement_score(samples) == 10.0


def test_movement_score_handles_empty_and_singletons():
    assert movement_score([]) == 0.0
    assert movement_score([{"shoulder_pan": 1.0}]) == 0.0  # need >=2 samples per joint


def test_decide_roles_picks_the_moved_arm_as_leader():
    out = decide_roles(40.0, "COM4", 1.0, "COM5")
    assert out == {"leader": "COM4", "follower": "COM5"}
    out2 = decide_roles(0.5, "COM4", 38.0, "COM5")
    assert out2 == {"leader": "COM5", "follower": "COM4"}


def test_decide_roles_ambiguous_when_both_move():
    # both moved a lot and within ratio -> can't tell them apart
    assert decide_roles(30.0, "COM4", 25.0, "COM5") is None


def test_decide_roles_none_when_nothing_moved():
    assert decide_roles(2.0, "COM4", 1.0, "COM5") is None  # below min_move_deg
