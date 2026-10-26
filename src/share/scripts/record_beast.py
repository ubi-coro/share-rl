import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any, Callable

import numpy as np
import torch
from beast_refiner.async_inference import AsyncPolicyRunner
from beast_refiner.utils.visualization import EpisodeRecorder, PolicyVisualizationAdapter
from lerobot.configs import parser
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    TransitionKey
)
from lerobot.utils.constants import ACTION, OBS_STATE, REWARD, DONE
from lerobot.utils.control_utils import predict_action
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from share.configs.record_beast import RecordBeastConfig
from share.debug.mpnet_debug import MPNetDebugger
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import ManipulationPrimitiveNet
from share.teleoperators import TeleopEvents
from share.utils.control_utils import (
    TickStats, episode_progress, make_async_runners, make_policies_and_datasets, mute_log_messages,
    offer_to_save_episode, predict_action_async
)
from share.utils.video_utils import MultiVideoEncodingManager

# INFO on the console. The async runner's per-plan timings and the loop's per-tick timings are DEBUG.
init_logging(console_level="INFO")
# The clamp itself stays active, only its per-tick warning is hidden
mute_log_messages("Relative goal position magnitude")

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
    recorders: dict[str, EpisodeRecorder] | None = None,
    runners: dict[str, AsyncPolicyRunner] | None = None,
    on_step: Callable[[dict], None] | None = None,
):
    runners = runners or {}

    # reset
    transition = mp_net.reset()
    if runners.get(mp_net.active_primitive, None) is not None:
        # also resets the policy, and drops the System 2 plan still in flight
        runners[mp_net.active_primitive].reset()
    elif policies.get(mp_net.active_primitive, None) is not None:
        policies[mp_net.active_primitive].reset()
    if debugger is not None:
        debugger.log_reset(mp_net, transition)

    # check if we need to terminate early, ie during reset
    info = transition.get(TransitionKey.INFO, {})
    if info.get(TeleopEvents.STOP_RECORDING, False):
        return info

    # get task description
    task = mp_net.config.primitives[mp_net.active_primitive].task_description
    task = mp_net.active_primitive if task is None else task

    # record loop
    sum_reward = 0.0
    # The async runners index System 1 against this: +1 per control step, restarting with every episode.
    obs_timestep = 0
    tick_stats = TickStats(window=round(mp_net.config.fps))
    while True:
        start_loop_t = time.perf_counter()
        obs = transition[TransitionKey.OBSERVATION]
        policy = policies.get(mp_net.active_primitive, None)
        runner = runners.get(mp_net.active_primitive, None)
        dataset = datasets.get(mp_net.active_primitive, None)
        phases = {}  # durations of the parts of this tick, summarised by tick_stats

        # (1) Decide and process action a_t
        if policy is not None:
            # Filter using the dataset's (pre-rename) feature schema, not the policy's
            # (post-rename) input_features -- obs still has the raw env/dataset key names,
            # and the preprocessor's rename_observations_processor is what maps them to the
            # names the policy expects. Filtering by policy.config.input_features here would
            # drop every renamed key (e.g. all camera images) before the rename step ever runs.
            obs_keys = dataset.features if dataset is not None else policy.config.input_features
            inference_kwargs = dict(
                observation={key: obs[key] for key in obs_keys if key in obs},
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessors[mp_net.active_primitive],
                postprocessor=postprocessors[mp_net.active_primitive],
                task=task,
                robot_type=mp_net.config.type
            )
            if runner is not None:
                # System 1 answers from the newest System 2 plan, System 2 replans on its own thread
                action = predict_action_async(runner=runner, obs_timestep=obs_timestep, **inference_kwargs)
            else:
                # noinspection PyTypeChecker
                action = predict_action(policy=policy, use_amp=policy.config.use_amp, **inference_kwargs)
            action = action.squeeze()
            policy_done_t = time.perf_counter()
            phases["policy"] = policy_done_t - start_loop_t

            # obs and action share the same absolute-joint-state space in this
            # env config (see AlohaBimanualMPNetConfig's TaskFrame), so the
            # currently sensed state can be plotted directly against the
            # policy's plans/executed action on the dashboard's "ground truth"
            # trace without any extra unit conversion.
            recorder = (recorders or {}).get(mp_net.active_primitive, None)
            if recorder is not None:
                recorder.record_step(
                    predicted_action=action,
                    ground_truth_action=obs.get(OBS_STATE)
                )
                phases["dashboard"] = time.perf_counter() - policy_done_t
        else:
            # Dummy action, expected to be overwritten by teleop action
            action = torch.tensor([0.0] * mp_net.action_dim, dtype=torch.float32)

        # (2) Step environment (sends the action, reads the next observation)
        step_start_t = time.perf_counter()
        try:
            new_transition = mp_net.step(action)
        except Exception as exc:
            # E.g. the arm drops the object and a control board disconnects. The robot cannot be stepped any more,
            # but the episode so far is worth keeping. The error still propagates, after the dataset is saved.
            logging.error(f"[{task}] env step failed: {exc!r}")
            offer_to_save_episode(dataset)
            raise
        phases["env_step"] = time.perf_counter() - step_start_t
        if debugger is not None:
            debugger.log_step(mp_net, new_transition)

        action = new_transition[TransitionKey.ACTION]
        reward = new_transition[TransitionKey.REWARD]
        done = new_transition.get(TransitionKey.DONE, False)
        truncated = new_transition.get(TransitionKey.TRUNCATED, False)
        info = new_transition.get(TransitionKey.INFO, {})
        sum_reward += float(reward)
        if on_step is not None:
            on_step(info)

        # (3) Exit on episode end
        if info.get(TeleopEvents.INTERVENTION_COMPLETED, False) or info.get(TeleopEvents.STOP_RECORDING, False):
            return info

        # (4) Store transition. When interactive, only store frames on interventions
        # store o_t, a_t, r_t+1
        if dataset is not None and (not interactive or info.get(TeleopEvents.IS_INTERVENTION, False)):
            store_start_t = time.perf_counter()
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
            phases["store"] = time.perf_counter() - store_start_t

        # (5) Update current observation
        transition = new_transition
        obs_timestep += 1

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
        logging.debug(
            "[%s] dt_loop: %5.2fms (%3.1fhz), dt_load: %5.2fms (%3.1fhz)",
            task, dt_loop * 1000, 1 / dt_loop, dt_load * 1000, 1 / dt_load
        )
        summary = tick_stats.add(dt_loop, dt_load, **phases)
        if summary is not None:
            logging.info(f"[{task}] {summary}")


@parser.wrap()
def record(cfg: RecordBeastConfig) -> LeRobotDataset:
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

    # Rollout dashboards are only recorded when there is somewhere to flush them to: an unflushed
    # recorder would keep every step's plan snapshot in memory for the whole session.
    output_dir = cfg.output_dir
    recorders = {
        name: EpisodeRecorder(
            adapter=PolicyVisualizationAdapter(policies[name], action_postprocessor=postprocessors[name])
        )
        for name in policies if policies[name] is not None and output_dir is not None
    }

    runners: dict[str, AsyncPolicyRunner] = {}
    try:
        runners = make_async_runners(policies, cfg.async_inference, cfg.env.fps)
        with MultiVideoEncodingManager(datasets):
            while True:
                log_say(f"Record episode for {mp_net.active_primitive}", play_sounds=cfg.play_sounds)

                dataset = datasets.get(mp_net.active_primitive, None)
                recorder = recorders.get(mp_net.active_primitive, None)
                recorder_root = None if output_dir is None else Path(output_dir) / mp_net.active_primitive

                with episode_progress(mp_net, dataset) as on_step:
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
                        recorders=recorders,
                        runners=runners,
                        on_step=on_step,
                    )

                if info.get(TeleopEvents.STOP_RECORDING, False):
                    break

                # Named after the dataset episode it belongs to: the recorder's own counter restarts at 0 in
                # every session, so a resumed run would overwrite the earlier dashboards. A re-recorded
                # episode reuses its index and replaces the dashboard of the discarded take.
                if recorder is not None and output_dir is not None:
                    # Named after the dataset episode it belongs to: the recorder's own counter restarts at 0 in
                    # every session, so a resumed run would overwrite the earlier dashboards. A re-recorded
                    # episode reuses its index and replaces the dashboard of the discarded take.
                    episode_name = None if dataset is None else f"episode_{dataset.num_episodes:06d}"
                    recorder.flush(output_root=recorder_root, episode_name=episode_name)

                if dataset is None:
                    continue

                # dataset ops, saving / clearing episode buffers
                if info.get(TeleopEvents.RERECORD_EPISODE, False):
                    log_say("Re-record episode", cfg.play_sounds, blocking=True)
                    dataset.clear_episode_buffer()
                elif dataset.writer.episode_buffer["size"] > 0:
                    # non-blocking: a blocking announcement delays the save by the length of the speech
                    log_say("Save episode", cfg.play_sounds)
                    dataset.save_episode()
                else:
                    log_say("Dataset is empty, continue execution", cfg.play_sounds, blocking=True)

    finally:
        for runner in runners.values():
            runner.stop()
        #debugger.close()
        log_say("Stop recording", cfg.play_sounds, blocking=True)
        mp_net.close()


if __name__ == "__main__":
    import experiments
    record()
