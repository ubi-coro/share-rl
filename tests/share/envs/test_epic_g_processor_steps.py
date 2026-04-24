"""Focused tests for observation processors and action flattening."""

from __future__ import annotations

import pytest
import torch

from lerobot.processor import TransitionKey
from share.envs.manipulation_primitive.task_frame import ControlMode, ControlSpace, PolicyMode, TaskFrame
from share.processor.action import RelativeFrameActionProcessor
from share.processor.observation import (
    JointsToEEObservation,
    RelativeFrameObservationProcessor,
    StateObservationProcessor,
    ImagePreprocessingProcessor,
)
from share.processor.utils import flatten_nested_policy_action
from share.utils.transformation_utils import euler_xyz_from_rotation, euler_xyz_from_rotvec, rotation_from_extrinsic_xyz
from tests.share.envs.mock_pipeline_entities import MockComplexKinematicsSolver


def _transition(action=None, observation=None):
    return {
        TransitionKey.OBSERVATION: observation or {},
        TransitionKey.ACTION: action,
        TransitionKey.REWARD: 0.0,
        TransitionKey.DONE: False,
        TransitionKey.TRUNCATED: False,
        TransitionKey.INFO: {},
        TransitionKey.COMPLEMENTARY_DATA: {},
    }


def test_joints_to_ee_observation_adds_expected_ee_pose_keys():
    """FK augmentation should append deterministic EE pose channels."""
    solver = MockComplexKinematicsSolver(joint_names=["joint_1", "joint_2", "joint_3"])
    step = JointsToEEObservation(kinematics={"arm": solver}, motor_names={"arm": ["joint_1", "joint_2", "joint_3"]})

    observation = {
        "arm.joint_1.pos": 0.35,
        "arm.joint_2.pos": -0.25,
        "arm.joint_3.pos": 0.55,
    }

    out = step(_transition(observation=observation))
    obs_out = out[TransitionKey.OBSERVATION]

    assert obs_out["arm.x.ee_pos"] == pytest.approx(0.5 * 0.35 + 0.2 * -0.25 - 0.1 * 0.55)
    assert obs_out["arm.y.ee_pos"] == pytest.approx(-0.3 * 0.35 + 0.4 * -0.25 + 0.2 * 0.55)
    assert obs_out["arm.z.ee_pos"] == pytest.approx(0.35 - (-0.25) + 0.55)
    assert obs_out["arm.rx.ee_pos"] == pytest.approx(0.1 * 0.35)
    assert obs_out["arm.ry.ee_pos"] == pytest.approx(-0.05 * -0.25)
    assert obs_out["arm.rz.ee_pos"] == pytest.approx(0.2 * 0.55)


def test_joints_to_ee_observation_raises_on_missing_joint_key():
    """Missing joint inputs should fail clearly."""
    step = JointsToEEObservation(
        kinematics={"arm": MockComplexKinematicsSolver()},
        motor_names={"arm": ["joint_1", "joint_2", "joint_3"]},
    )

    with pytest.raises(ValueError, match="Missing joint observation key 'arm.joint_3.pos'"):
        step(_transition(observation={"arm.joint_1.pos": 0.1, "arm.joint_2.pos": 0.2}))


def test_relative_frame_observation_processor_tracks_per_robot_reference():
    """Relative EE pose should be measured from each robot's stored reference pose."""
    step = RelativeFrameObservationProcessor(enable={"arm": True, "other": False})

    first = _transition(
        observation={
            "arm.x.ee_pos": 1.0,
            "arm.y.ee_pos": 2.0,
            "arm.z.ee_pos": 3.0,
            "arm.rx.ee_pos": 0.1,
            "arm.ry.ee_pos": 0.2,
            "arm.rz.ee_pos": 0.3,
            "other.x.ee_pos": 5.0,
            "other.y.ee_pos": 6.0,
            "other.z.ee_pos": 7.0,
            "other.rx.ee_pos": 0.5,
            "other.ry.ee_pos": 0.6,
            "other.rz.ee_pos": 0.7,
        }
    )
    out1 = step(first)[TransitionKey.OBSERVATION]
    assert out1["arm.x.ee_pos"] == pytest.approx(0.0)
    assert out1["arm.rz.ee_pos"] == pytest.approx(0.0)
    assert out1["other.x.ee_pos"] == pytest.approx(5.0)

    second = _transition(
        observation={
            "arm.x.ee_pos": 1.5,
            "arm.y.ee_pos": 1.0,
            "arm.z.ee_pos": 4.0,
            "arm.rx.ee_pos": 0.2,
            "arm.ry.ee_pos": -0.2,
            "arm.rz.ee_pos": 0.4,
        }
    )
    out2 = step(second)[TransitionKey.OBSERVATION]
    assert out2["arm.x.ee_pos"] == pytest.approx(0.5)
    assert out2["arm.y.ee_pos"] == pytest.approx(-1.0)
    assert out2["arm.z.ee_pos"] == pytest.approx(1.0)
    expected_orientation = euler_xyz_from_rotation(
        rotation_from_extrinsic_xyz(*euler_xyz_from_rotvec([0.2, -0.2, 0.4])) *
        rotation_from_extrinsic_xyz(*euler_xyz_from_rotvec([0.1, 0.2, 0.3])).inv()
    )
    assert out2["arm.rx.ee_pos"] == pytest.approx(expected_orientation[0])
    assert out2["arm.ry.ee_pos"] == pytest.approx(expected_orientation[1])
    assert out2["arm.rz.ee_pos"] == pytest.approx(expected_orientation[2])


def test_relative_frame_observation_processor_reset_reinitializes_reference():
    """Reset should clear the stored reference pose."""
    step = RelativeFrameObservationProcessor(enable=True)

    step(
        _transition(
            observation={
                "arm.x.ee_pos": 1.0,
                "arm.y.ee_pos": 2.0,
                "arm.z.ee_pos": 3.0,
                "arm.rx.ee_pos": 0.1,
                "arm.ry.ee_pos": 0.2,
                "arm.rz.ee_pos": 0.3,
            }
        )
    )
    step.reset()
    out = step(
        _transition(
            observation={
                "arm.x.ee_pos": -2.0,
                "arm.y.ee_pos": -3.0,
                "arm.z.ee_pos": -4.0,
                "arm.rx.ee_pos": -0.1,
                "arm.ry.ee_pos": -0.2,
                "arm.rz.ee_pos": -0.3,
            }
        )
    )[TransitionKey.OBSERVATION]

    assert out["arm.x.ee_pos"] == pytest.approx(0.0)
    assert out["arm.rz.ee_pos"] == pytest.approx(0.0)


def test_relative_frame_action_processor_transforms_kinematic_axes_only():
    """The current relative-frame action step is a numeric no-op."""
    step = RelativeFrameActionProcessor(enable={"arm": True})
    action = {
        "joint_1.pos": 0.1,
        "joint_2.pos": -0.2,
        "gripper.pos": 0.75,
    }
    out = step(_transition(action=action))[TransitionKey.ACTION]
    assert out["joint_1.pos"] == pytest.approx(0.1)
    assert out["joint_2.pos"] == pytest.approx(-0.2)
    assert out["gripper.pos"] == pytest.approx(0.75)


def test_relative_frame_action_processor_is_noop_when_disabled():
    """Disabling the step should leave actions untouched."""
    step = RelativeFrameActionProcessor(enable=False)
    action = {"joint_1.pos": 0.2}
    out = step(_transition(action=action))[TransitionKey.ACTION]
    assert out == action


def test_flatten_nested_policy_action_uses_task_frame_learning_order():
    """Flattening should follow each frame's learning-space key order."""
    action = {
        "arm": {
            "x.ee_pos": 1.0,
            "rotation.so3.a1.x": 2.0,
            "rotation.so3.a1.y": 3.0,
            "rotation.so3.a1.z": 4.0,
            "rotation.so3.a2.x": 5.0,
            "rotation.so3.a2.y": 6.0,
            "rotation.so3.a2.z": 7.0,
        },
        "wrist": {
            "joint_a.pos": 8.0,
        },
    }
    task_frame = {
        "arm": TaskFrame(
            policy_mode=[
                PolicyMode.ABSOLUTE,
                None,
                None,
                PolicyMode.ABSOLUTE,
                PolicyMode.ABSOLUTE,
                PolicyMode.ABSOLUTE,
            ],
            control_mode=[ControlMode.POS] * 6,
        ),
        "wrist": TaskFrame(
            target=[0.0],
            space=ControlSpace.JOINT,
            policy_mode=[PolicyMode.ABSOLUTE],
            control_mode=[ControlMode.POS],
            joint_names=["joint_a"],
        ),
    }

    out = flatten_nested_policy_action(
        action,
        task_frame=task_frame,
        gripper_enable={"arm": False, "wrist": False},
    )
    torch.testing.assert_close(out, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))


def test_flatten_nested_policy_action_raises_on_missing_key():
    """Missing keyed entries should fail loudly."""
    frame = TaskFrame(
        policy_mode=[PolicyMode.ABSOLUTE, None, None, None, None, None],
        control_mode=[ControlMode.POS] * 6,
    )

    with pytest.raises(ValueError, match="Missing policy action key 'arm.x.ee_pos'"):
        flatten_nested_policy_action(
            {"arm": {}},
            task_frame={"arm": frame},
            gripper_enable={"arm": False},
        )


def test_default_observation_processor_collects_modalities_and_images():
    """Enabled modalities should populate state and normalize images."""
    step = StateObservationProcessor(
        gripper_enable={"arm": True},
        add_joint_position_to_observation={"arm": True},
        add_joint_velocity_to_observation={"arm": True},
        add_current_to_observation={"arm": True},
        add_ee_pos_to_observation={"arm": True},
        add_ee_velocity_to_observation={"arm": True},
        add_ee_wrench_to_observation={"arm": True},
    )

    observation = {
        "arm.joint_1.pos": 1.0,
        "arm.joint_2.pos": 2.0,
        "arm.joint_1.current": 0.1,
        "arm.joint_2.current": 0.2,
        "arm.x.ee_pos": 0.01,
        "arm.y.ee_pos": 0.02,
        "arm.z.ee_pos": 0.03,
        "arm.rx.ee_pos": 0.04,
        "arm.ry.ee_pos": 0.05,
        "arm.rz.ee_pos": 0.06,
        "arm.x.ee_wrench": 1.0,
        "arm.y.ee_wrench": 2.0,
        "arm.z.ee_wrench": 3.0,
        "arm.rx.ee_wrench": 4.0,
        "arm.ry.ee_wrench": 5.0,
        "arm.rz.ee_wrench": 6.0,
        "arm.gripper.pos": 0.9,
        "observation.images.cam": torch.full((8, 8, 3), 255, dtype=torch.uint8),
    }

    first = step(_transition(observation=observation))[TransitionKey.OBSERVATION]
    assert first["observation.images.cam"].shape == (3, 8, 8)
    assert first["observation.images.cam"].dtype == torch.float32

    state = first["observation.state"]
    # The current processor includes joint pos/vel/current, ee_pos, ee_wrench, and gripper.
    # EE velocity fallback is not populated from pose channels in the current implementation.
    assert state.shape == (19,)
    torch.testing.assert_close(state[2:4], torch.zeros(2))
    torch.testing.assert_close(state, torch.tensor([
        1.0,
        2.0,
        0.0,
        0.0,
        0.1,
        0.2,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        0.9,
    ]))


def test_image_preprocessing_processor_crops_before_normalizing_images():
    """Image preprocessing should apply camera-key crops and return CHW floats."""
    step = ImagePreprocessingProcessor(
        crop_params_dict={"cam": (1, 1, 2, 2)},
        resize_size=None,
    )

    image = torch.arange(4 * 4 * 3, dtype=torch.uint8).reshape(4, 4, 3)
    out = step(_transition(observation={"observation.images.cam": image}))[TransitionKey.OBSERVATION]

    expected = image[1:3, 1:3, :].permute(2, 0, 1).to(torch.float32) / 255.0
    assert out["observation.images.cam"].shape == (3, 2, 2)
    assert out["observation.images.cam"].dtype == torch.float32
    torch.testing.assert_close(out["observation.images.cam"], expected)


def test_default_observation_processor_transform_features_counts_enabled_modalities():
    """Feature inference should reflect enabled state channels."""
    step = StateObservationProcessor(
        gripper_enable={"arm": True},
        add_joint_position_to_observation={"arm": True},
        add_joint_velocity_to_observation={"arm": True},
        add_current_to_observation={"arm": False},
        add_ee_pos_to_observation={"arm": True},
        add_ee_velocity_to_observation={"arm": True},
        add_ee_wrench_to_observation={"arm": False},
    )

    from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature

    features = {
        PipelineFeatureType.OBSERVATION: {
            "arm.joint_1.pos": PolicyFeature(type=FeatureType.STATE, shape=(1,)),
            "arm.joint_2.pos": PolicyFeature(type=FeatureType.STATE, shape=(1,)),
            "arm.x.ee_pos": PolicyFeature(type=FeatureType.STATE, shape=(1,)),
            "arm.y.ee_pos": PolicyFeature(type=FeatureType.STATE, shape=(1,)),
            "arm.z.ee_pos": PolicyFeature(type=FeatureType.STATE, shape=(1,)),
            "arm.rx.ee_pos": PolicyFeature(type=FeatureType.STATE, shape=(1,)),
            "arm.ry.ee_pos": PolicyFeature(type=FeatureType.STATE, shape=(1,)),
            "arm.rz.ee_pos": PolicyFeature(type=FeatureType.STATE, shape=(1,)),
            "arm.gripper.pos": PolicyFeature(type=FeatureType.STATE, shape=(1,)),
        },
        PipelineFeatureType.ACTION: {},
    }

    out = step.transform_features(features)
    assert out[PipelineFeatureType.OBSERVATION]["observation.state"].shape == (17,)


def test_default_observation_processor_supports_axis_selection_and_frame_stacking():
    """Axis filtering and frame stacking should shape observation.state consistently."""

    step = StateObservationProcessor(
        add_joint_position_to_observation={"arm": False},
        add_joint_velocity_to_observation={"arm": False},
        add_current_to_observation={"arm": False},
        add_ee_pos_to_observation={"arm": True},
        ee_pos_axes={"arm": ["z.ee_pos"]},
        add_ee_velocity_to_observation={"arm": False},
        add_ee_wrench_to_observation={"arm": True},
        ee_wrench_axes={"arm": ["x.ee_wrench", "y.ee_wrench", "z.ee_wrench"]},
        stack_frames={"arm": 2},
    )

    observation = {
        "arm.x.ee_pos": 0.01,
        "arm.y.ee_pos": 0.02,
        "arm.z.ee_pos": 0.03,
        "arm.rx.ee_pos": 0.04,
        "arm.ry.ee_pos": 0.05,
        "arm.rz.ee_pos": 0.06,
        "arm.x.ee_wrench": 1.0,
        "arm.y.ee_wrench": 2.0,
        "arm.z.ee_wrench": 3.0,
    }

    first = step(_transition(observation=observation))[TransitionKey.OBSERVATION]["observation.state"]
    second_obs = dict(observation)
    second_obs["arm.z.ee_pos"] = 0.05
    second = step(_transition(observation=second_obs))[TransitionKey.OBSERVATION]["observation.state"]

    assert first.shape == (8,)
    torch.testing.assert_close(first, torch.tensor([0.03, 1.0, 2.0, 3.0, 0.03, 1.0, 2.0, 3.0]))
    torch.testing.assert_close(second, torch.tensor([0.03, 1.0, 2.0, 3.0, 0.05, 1.0, 2.0, 3.0]))
