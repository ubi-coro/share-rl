import logging
import time
from dataclasses import asdict
from pprint import pformat
from typing import Any

import numpy as np
import torch
from lerobot.configs import parser
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    TransitionKey
)
from lerobot.utils.constants import ACTION, REWARD, DONE
from lerobot.utils.control_utils import predict_action
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

try:
    from lerobot.utils.device_utils import get_safe_torch_device
except ImportError:  # pragma: no cover - compatibility with older lerobot layouts
    from lerobot.utils.utils import get_safe_torch_device

from share.configs.record import RecordConfig
from share.debug.mpnet_debug import MPNetDebugger
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import ManipulationPrimitiveNet
from share.teleoperators import TeleopEvents, has_event, is_intervention
from share.utils.control_utils import make_policies_and_datasets
from share.utils.logging_utils import log_runtime_frequency
from share.utils.video_utils import MultiVideoEncodingManager

init_logging()

""" --------------- record_loop() data flow --------------------------
       [ Robot ]
           V
     [ robot.get_observation() ] ---> raw_obs
           V
     [ robot_observation_processor ] ---> processed_obs
           V
     .-----( ACTION LOGIC )------------------.
     V                                       V
     [ From Teleoperator ]                   [ From Policy ]
     |                                       |
     |  [teleop.get_action] -> raw_action    |   [predict_action]
     |          |                            |          |
     |          V                            |          V
     | [teleop_action_processor]             |          |
     |          |                            |          |
     '---> processed_teleop_action           '---> processed_policy_action
     |                                       |
     '-------------------------.-------------'
                               V
                  [ robot_action_processor ] --> robot_action_to_send
                               V
                    [ robot.send_action() ] -- (Robot Executes)
                               V
                    ( Save to Dataset )
                               V
                  ( Rerun Log / Loop Wait )
"""

@safe_stop_image_writer
def record_loop(
    mp_net: ManipulationPrimitiveNet,
    datasets: dict[str, LeRobotDataset],
    policies: dict[str, PreTrainedPolicy],
    preprocessors: dict[str, PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]],
    postprocessors: dict[str, PolicyProcessorPipeline[PolicyAction, PolicyAction]],
    display_data: bool = False,
    display_compressed_images: bool = False,
    save_only_interventions: bool = False,
    force_intervention: bool = False,
    debugger: MPNetDebugger | None = None,
):
    # reset
    transition = mp_net.reset()
    policy = policies.get(mp_net.active_primitive, None)
    if policy is not None:
        policies[mp_net.active_primitive].reset()
    if debugger is not None:
        debugger.log_reset(mp_net, transition)

    # check if we need to terminate early, ie during reset
    info = transition.get(TransitionKey.INFO, {})
    if has_event(info, TeleopEvents.STOP_RECORDING):
        return info

    # get task description
    task = mp_net.config.primitives[mp_net.active_primitive].task_description
    task = mp_net.active_primitive if task is None else task

    # record loop
    sum_reward = 0.0
    while True:
        start_loop_t = time.perf_counter()
        obs = transition[TransitionKey.OBSERVATION]
        policy = policies.get(mp_net.active_primitive, None)
        dataset = datasets.get(mp_net.active_primitive, None)

        # (1) Decide and process action a_t
        if policy is not None and not force_intervention:
            # noinspection PyTypeChecker
            action = predict_action(
                observation={key: obs[key] for key in policy.config.input_features},
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessors[mp_net.active_primitive],
                postprocessor=postprocessors[mp_net.active_primitive],
                use_amp=policy.config.use_amp,
                task=task,
                robot_type=mp_net.config.type
            ).squeeze()
        else:
            # Dummy action, expected to be overwritten by teleop action
            action = torch.tensor([0.0] * mp_net.action_dim, dtype=torch.float32)

        # (2) Step environment
        new_transition = mp_net.step(action)
        if debugger is not None:
            debugger.log_step(mp_net, new_transition)

        action = new_transition[TransitionKey.ACTION]
        reward = new_transition[TransitionKey.REWARD]
        done = new_transition.get(TransitionKey.DONE, False)
        truncated = new_transition.get(TransitionKey.TRUNCATED, False)
        info = new_transition.get(TransitionKey.INFO, {})
        sum_reward += float(reward)

        # (3) Exit on episode end
        intervention_segment_finished = (
            policy is not None and
            save_only_interventions and
            has_event(info, TeleopEvents.INTERVENTION_COMPLETED)
        )
        if intervention_segment_finished or has_event(info, TeleopEvents.STOP_RECORDING):
            return info

        # (4) Store transition. In correction-only mode, only intervention steps are saved.
        # store o_t, a_t, r_t+1
        if dataset is not None and (not save_only_interventions or is_intervention(info)):
            # observations are batched and may contain other keys
            dataset_observation = {
                k: v.squeeze().cpu()
                for k, v in obs.items()
                if k in dataset.features
            }

            # store frame
            frame = {
                **dataset_observation,
                ACTION: action.squeeze().cpu(),
                REWARD: np.array([reward], dtype=np.float32),
                DONE: np.array([done], dtype=bool),
                "task": task
            }
            dataset.add_frame(frame)

            if display_data:
                rerun_obs = {k: v.numpy() for k, v in dataset_observation.items()}
                log_rerun_data(
                    observation=rerun_obs, action=action.squeeze().cpu(), compress_images=display_compressed_images
                )

        # (5) Update current observation
        transition = new_transition

        # 6) Handle done
        if (
            done or
            truncated or
            has_event(info, TeleopEvents.RERECORD_EPISODE)
        ):
            return info

        # (7) Handle frequency
        dt_load = time.perf_counter() - start_loop_t
        precise_sleep(1 / mp_net.config.fps - dt_load)
        dt_loop = time.perf_counter() - start_loop_t
        log_runtime_frequency(
            prefix="RECORD",
            primitive=mp_net.active_primitive,
            task=task,
            loop_dt_s=dt_loop,
            work_dt_s=dt_load,
            work_label="step",
        )


@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    # make
    mp_net = ManipulationPrimitiveNet(cfg.env)
    force_intervention = not cfg.use_policy
    mp_net.set_step_info({TeleopEvents.IS_INTERVENTION: True} if force_intervention else None)
    debugger = None
    datasets, policies, preprocessors, postprocessors = make_policies_and_datasets(cfg)

    try:
        with MultiVideoEncodingManager(datasets):
            while True:
                log_say(f"Record episode for {mp_net.active_primitive}", play_sounds=cfg.play_sounds)

                dataset = datasets.get(mp_net.active_primitive, None)

                info = record_loop(
                    mp_net=mp_net,
                    datasets=datasets,
                    policies=policies,
                    preprocessors=preprocessors,
                    postprocessors=postprocessors,
                    display_data=cfg.display_data,
                    display_compressed_images=display_compressed_images,
                    save_only_interventions=cfg.save_only_interventions,
                    force_intervention=force_intervention,
                    debugger=debugger,
                )

                if has_event(info, TeleopEvents.STOP_RECORDING):
                    break

                if dataset is None:
                    continue

                # dataset ops, saving / clearing episode buffers
                if has_event(info, TeleopEvents.RERECORD_EPISODE):
                    log_say("Re-record episode", cfg.play_sounds, blocking=True)
                    dataset.clear_episode_buffer()
                elif dataset.writer.episode_buffer["size"] > 0:
                    log_say("Save episode", cfg.play_sounds, blocking=True)
                    dataset.save_episode()
                else:
                    log_say("Dataset is empty, continue execution", cfg.play_sounds, blocking=True)
    finally:
        #debugger.close()
        log_say("Stop recording", cfg.play_sounds, blocking=True)
        mp_net.close()


if __name__ == "__main__":
    import experiments
    record()
