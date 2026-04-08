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
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from share.configs.record import RecordConfig
from share.debug.mpnet_debug import MPNetDebugger
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import ManipulationPrimitiveNet
from share.teleoperators import TeleopEvents
from share.utils.control_utils import make_policies_and_datasets
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
    interactive: bool = False,
    debugger: MPNetDebugger | None = None,
):
    transition = mp_net.reset()
    if debugger is not None:
        debugger.log_reset(mp_net, transition)

    # check if we need to terminate early
    info = transition.get(TransitionKey.INFO, {})
    if info.get(TeleopEvents.STOP_RECORDING, False):
        return info

    # get task description
    task = mp_net.config.primitives[mp_net.active_primitive].task_description
    task = mp_net.active_primitive if task is None else task

    sum_reward = 0.0
    while True:
        start_loop_t = time.perf_counter()
        obs = transition[TransitionKey.OBSERVATION]
        policy = policies.get(mp_net.active_primitive, None)
        dataset = datasets.get(mp_net.active_primitive, None)

        # (1) Decide and process action a_t
        if policy is not None:
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
        if info.get(TeleopEvents.INTERVENTION_COMPLETED, False) or info.get(TeleopEvents.STOP_RECORDING, False):
            return info

        # (4) Store transition. When interactive, only store frames on interventions
        # store o_t, a_t, r_t+1
        if dataset is not None and (not interactive or info.get(TeleopEvents.IS_INTERVENTION, False)):
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
            info.get(TeleopEvents.RERECORD_EPISODE, False)
        ):
            return info

        # (7) Handle frequency
        dt_load = time.perf_counter() - start_loop_t
        precise_sleep(1 / mp_net.config.fps - dt_load)
        dt_loop = time.perf_counter() - start_loop_t
        logging.info(
            f"[{task}] "
            #f"{transition[TransitionKey.OBSERVATION]["main.x.ee_pos"]}, "
            #f"{transition[TransitionKey.OBSERVATION]["main.y.ee_pos"]}, "
            #f"{transition[TransitionKey.OBSERVATION]["main.z.ee_pos"]}"
            f"dt_loop: {dt_loop * 1000:5.2f}ms ({1 / dt_loop:3.1f}hz), "
            f"dt_load: {dt_load * 1000:5.2f}ms ({1 / dt_load:3.1f}hz)"
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
                    interactive=cfg.interactive,
                    debugger=debugger,
                )

                if info.get(TeleopEvents.STOP_RECORDING, False):
                    break

                if dataset is None:
                    continue

                # dataset ops, saving / clearing episode buffers
                if info.get(TeleopEvents.RERECORD_EPISODE, False):
                    log_say("Re-record episode", cfg.play_sounds, blocking=True)
                    dataset.clear_episode_buffer()
                elif dataset.episode_buffer["size"] > 0:
                    log_say("Save episode", cfg.play_sounds, blocking=False)
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
