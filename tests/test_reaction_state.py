import numpy as np
import pytest

from lerobot_policy_spacepolicy_act.reaction_state import (
    LEFT15_DIM,
    REACTION_MODEL_STATE_DIM,
    Left15Command,
    RobotState,
    TaskState,
    pack_reaction_model_state,
    relative_base_pose_body_xyzw,
)


def _identity_pose(position=(0.0, 0.0, 0.0)):
    return np.array([*position, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def test_robot_state_preserves_full_measured_q_dq_and_matches_online_base_projection():
    state = RobotState.from_canonical_diagnostics(
        base_pose=_identity_pose((1.0, 2.0, 3.0)),
        base_velocity=np.arange(6, dtype=np.float32) + 10.0,
        raw_joint_state=np.arange(26, dtype=np.float32),
        measured_joint_velocity=np.arange(26, dtype=np.float32) + 100.0,
    )

    np.testing.assert_array_equal(state.joint_position, np.arange(26, dtype=np.float32))
    np.testing.assert_array_equal(
        state.joint_velocity,
        np.arange(26, dtype=np.float32) + 100.0,
    )
    np.testing.assert_array_equal(
        state.project_base_condition(output_dim=15),
        np.array([1, 2, 3, 1, 0, 0, 0, 1, 0, 10, 11, 12, 13, 14, 15], dtype=np.float32),
    )
    assert not state.joint_position.flags.writeable
    with pytest.raises(ValueError, match="NaN/Inf"):
        RobotState(_identity_pose(), np.zeros(6), np.zeros(26), np.r_[np.zeros(25), np.nan])


def test_body_frame_projections_are_explicit_and_quaternion_safe():
    half = np.sqrt(0.5)
    z_90_pose = np.array([0, 0, 0, 0, 0, half, half], dtype=np.float32)
    state = RobotState(z_90_pose, np.array([1, 0, 0, 0, 0, 0], dtype=np.float32), np.zeros(26), np.zeros(26))

    np.testing.assert_allclose(state.base_twist_body, [0, -1, 0, 0, 0, 0], atol=1e-6)
    relative = relative_base_pose_body_xyzw(z_90_pose, _identity_pose((1.0, 0.0, 0.0)))
    np.testing.assert_allclose(relative[:3], [0, -1, 0], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(relative[3:]), 1.0, atol=1e-6)
    assert not relative.flags.writeable


def test_left15_and_task_diagnostics_keep_semantics_separate_from_robot_features():
    command = Left15Command(np.arange(LEFT15_DIM, dtype=np.float32))
    np.testing.assert_array_equal(command.arm_position_target, np.arange(7, dtype=np.float32))
    np.testing.assert_array_equal(command.arm_desired_velocity, np.arange(7, 14, dtype=np.float32))
    np.testing.assert_array_equal(command.gripper_position_target, [14.0])

    pose = _identity_pose()
    task = TaskState.from_canonical_diagnostics(
        relative_target_pose=pose,
        delta_ee_target=pose,
        ee_pose=np.concatenate((pose, pose + np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float32))),
        grasp_success=0.0,
    )
    np.testing.assert_array_equal(task.left_ee_pose_world_xyzw, pose)
    np.testing.assert_array_equal(task.right_ee_pose_world_xyzw[:3], [1, 0, 0])
    assert task.grasp_success == 0.0


def test_reaction_state_pack_has_the_single_shared_80d_layout():
    robot = RobotState(_identity_pose(), np.arange(6, dtype=np.float32), np.arange(26, dtype=np.float32), np.arange(26, dtype=np.float32) + 30)
    packed = pack_reaction_model_state(robot, Left15Command(np.arange(15, dtype=np.float32) + 60))

    assert packed.shape == (REACTION_MODEL_STATE_DIM,)
    np.testing.assert_array_equal(packed[:7], _identity_pose())
    np.testing.assert_array_equal(packed[7:13], np.arange(6, dtype=np.float32))
    np.testing.assert_array_equal(packed[13:39], np.arange(26, dtype=np.float32))
    np.testing.assert_array_equal(packed[39:65], np.arange(26, dtype=np.float32) + 30)
    np.testing.assert_array_equal(packed[65:], np.arange(15, dtype=np.float32) + 60)
    assert not packed.flags.writeable
