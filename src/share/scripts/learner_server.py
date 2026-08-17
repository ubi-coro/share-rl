#!/usr/bin/env python

import logging
import os
import queue
import shutil
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import grpc
import torch
from torch.multiprocessing import Queue
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.rl.buffer import ReplayBuffer, concatenate_batch_transitions
from lerobot.rl.learner import check_nan_in_transition, get_observation_features
from lerobot.rl.learner_service import LearnerService, MAX_WORKERS, SHUTDOWN_TIMEOUT
from lerobot.rl.process import ProcessSignalHandler
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.transport import services_pb2_grpc
from lerobot.transport.utils import MAX_MESSAGE_SIZE, bytes_to_python_object, bytes_to_transitions, state_to_bytes
from lerobot.utils.constants import ACTION, DONE, REWARD, TRAINING_STATE_DIR
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    get_step_checkpoint_dir,
    load_training_state as utils_load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.transition import move_state_dict_to_device, move_transition_to_device
from lerobot.utils.utils import init_logging

from share.configs.rl import MPNetTrainRLServerPipelineConfig
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import (
    ManipulationPrimitiveNet,
)
from share.policies.sac_dagger import SACDaggerBCPolicy
from share.rl.runtime import (
    build_adaptive_registry,
    make_policy_processors,
    make_policies_for_registry,
    preprocess_replay_batch,
)
from share.utils.control_utils import suppress_logging
from share.utils.device import get_safe_torch_device
from share.utils.logging_utils import primary_loss


@parser.wrap()
def train_cli(cfg: MPNetTrainRLServerPipelineConfig):
    cfg.validate(output_role="learner")
    run_learner(cfg)


def run_learner(cfg: MPNetTrainRLServerPipelineConfig, shutdown_event: Any | None = None) -> dict[str, Any]:
    registry = build_adaptive_registry(cfg.env)
    is_threaded = _use_threads(registry.actor_learner_policy_cfg)

    if not is_threaded:
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")

    if shutdown_event is None:
        shutdown_event = ProcessSignalHandler(is_threaded, display_pid=not is_threaded).shutdown_event

    log_dir = os.path.join(str(cfg.output_dir), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"learner_{cfg.job_name}.log")
    init_logging(log_file=log_file, display_pid=not is_threaded)
    logging.info("Learner logging initialized, writing to %s", log_file)

    wandb_logger: WandBLogger | None = None
    if not cfg.wandb.enable:
        logging.info("[LEARNER] wandb disabled (cfg.wandb.enable=False); no run will be created.")
    elif not cfg.wandb.project:
        logging.warning("[LEARNER] wandb enabled but cfg.wandb.project is empty; skipping wandb run.")
    else:
        try:
            # WandBLogger calls wandb.init(config=cfg.to_dict()); to_dict() cannot encode the
            # live robot handles carried on cfg.env, so clear them for the duration exactly as
            # checkpoint serialization does. Guard the whole thing so a wandb failure degrades
            # to "train without logging" instead of silently leaving no run.
            with _checkpoint_safe_runtime_config(cfg):
                wandb_logger = WandBLogger(cfg)
        except Exception as exc:  # noqa: BLE001
            logging.exception("[LEARNER] Failed to initialize wandb run, continuing without it: %s", exc)
            wandb_logger = None

    set_seed(cfg.seed)

    result = start_learner_threads(
        cfg=cfg,
        registry=registry,
        shutdown_event=shutdown_event,
        wandb_logger=wandb_logger,
    )
    return result


def start_learner_threads(
    cfg: MPNetTrainRLServerPipelineConfig,
    registry: Any,
    shutdown_event: Any,
    wandb_logger: WandBLogger | None,
) -> dict[str, Any]:
    transition_queue = Queue()
    interaction_message_queue = Queue()
    parameters_queue = Queue()

    is_threaded = _use_threads(registry.actor_learner_policy_cfg)
    if is_threaded:
        from threading import Thread as ConcurrencyEntity
    else:
        from torch.multiprocessing import Process as ConcurrencyEntity

    communication_worker = ConcurrencyEntity(
        target=start_learner_server,
        args=(
            registry,
            shutdown_event,
            parameters_queue,
            transition_queue,
            interaction_message_queue,
        ),
        daemon=True,
    )
    communication_worker.start()

    try:
        result = add_actor_information_and_train(
            cfg=cfg,
            registry=registry,
            shutdown_event=shutdown_event,
            transition_queue=transition_queue,
            interaction_message_queue=interaction_message_queue,
            parameters_queue=parameters_queue,
            wandb_logger=wandb_logger,
        )
    finally:
        shutdown_event.set()
        communication_worker.join()

        transition_queue.close()
        interaction_message_queue.close()
        parameters_queue.close()

        transition_queue.cancel_join_thread()
        interaction_message_queue.cancel_join_thread()
        parameters_queue.cancel_join_thread()

    return result


def add_actor_information_and_train(
    cfg: MPNetTrainRLServerPipelineConfig,
    registry: Any,
    shutdown_event: Any,
    transition_queue: Queue,
    interaction_message_queue: Queue,
    parameters_queue: Queue,
    wandb_logger: WandBLogger | None,
) -> dict[str, Any]:
    device = str(get_safe_torch_device(registry.actor_learner_policy_cfg.device, log=True))
    storage_device = str(get_safe_torch_device(registry.actor_learner_policy_cfg.storage_device))

    with suppress_logging():
        mp_net = ManipulationPrimitiveNet(cfg.env)
        try:
            policies = make_policies_for_registry(cfg.env, registry, train_mode=True)
            # Reuse the exact same policy preprocessor stack as the actor so replay
            # training and live inference see identically normalized inputs.
            preprocessors, postprocessors = make_policy_processors(policies)
        finally:
            mp_net.close()

    # The env above connected to the robot/cameras only to read feature shapes, and closing it
    # frees the RTDE registers. The actor grabs the same robot, so it must not start until this
    # point -- tools/setup_connector.sh waits for this marker before launching the actor. Keep
    # the token in sync with that script.
    logging.info("[LEARNER] ROBOT_RELEASED: features captured, robot free for the actor.")

    optimizers = {primitive_id: make_optimizers(policy) for primitive_id, policy in policies.items()}
    resume_optimization_steps, resume_interaction_steps = load_training_state(
        cfg=cfg,
        optimizers=optimizers,
    )
    replay_buffers = initialize_replay_buffers(
        cfg=cfg,
        policies=policies,
        device=device,
        storage_device=storage_device,
    )
    offline_replay_buffers = initialize_offline_replay_buffers(
        cfg=cfg,
        policies=policies,
        device=device,
        storage_device=storage_device,
    )

    online_iterators: dict[str, Any] = {primitive_id: None for primitive_id in registry.adaptive_ids}
    offline_iterators: dict[str, Any] = {primitive_id: None for primitive_id in registry.adaptive_ids}
    optimization_steps = resume_optimization_steps or {primitive_id: 0 for primitive_id in registry.adaptive_ids}
    interaction_step_offset = resume_interaction_steps or {primitive_id: 0 for primitive_id in registry.adaptive_ids}
    last_interaction_messages: dict[str, dict[str, Any]] = {}

    push_all_actor_policies_to_queue(parameters_queue, policies)
    last_push_t = time.time()
    push_period_s = registry.actor_learner_policy_cfg.actor_learner_config.policy_parameters_push_frequency

    while not shutdown_event.is_set() and not _all_finished(optimization_steps, registry.online_step_budgets):
        process_transitions(
            transition_queue=transition_queue,
            replay_buffers=replay_buffers,
            offline_replay_buffers=offline_replay_buffers,
            device=device,
            shutdown_event=shutdown_event,
        )
        process_interaction_messages(
            interaction_message_queue=interaction_message_queue,
            shutdown_event=shutdown_event,
            wandb_logger=wandb_logger,
            interaction_step_offset=interaction_step_offset,
            last_messages=last_interaction_messages,
        )

        did_optimize = False
        for primitive_id in registry.adaptive_ids:
            policy = policies[primitive_id]
            if optimization_steps[primitive_id] >= registry.online_step_budgets[primitive_id]:
                continue

            replay_buffer = replay_buffers[primitive_id]
            is_dagger_bc_policy = isinstance(policy, SACDaggerBCPolicy)
            offline_replay_buffer = offline_replay_buffers.get(primitive_id)

            if is_dagger_bc_policy:
                if offline_replay_buffer is None or len(offline_replay_buffer) == 0:
                    continue
            elif len(replay_buffer) < policy.config.online_step_before_learning:
                continue

            online_batch_size = cfg.batch_size
            if offline_replay_buffer is not None and len(offline_replay_buffer) > 0 and not is_dagger_bc_policy:
                online_batch_size = max(1, cfg.batch_size // 2)

            if not is_dagger_bc_policy and online_iterators[primitive_id] is None:
                online_iterators[primitive_id] = replay_buffer.get_iterator(
                    batch_size=online_batch_size,
                    async_prefetch=policy.config.async_prefetch,
                    queue_size=2,
                )
            if offline_replay_buffer is not None and len(offline_replay_buffer) > 0 and offline_iterators[primitive_id] is None:
                offline_batch_size = cfg.batch_size if is_dagger_bc_policy else max(1, cfg.batch_size - online_batch_size)
                if is_dagger_bc_policy:
                    offline_iterators[primitive_id] = offline_replay_buffer.get_iterator(
                        batch_size=offline_batch_size,
                        async_prefetch=policy.config.async_prefetch,
                        queue_size=2,
                    )
                elif _num_td_valid_offline_samples(offline_replay_buffer) > 0:
                    offline_iterators[primitive_id] = _td_valid_offline_iterator(
                        offline_replay_buffer,
                        batch_size=offline_batch_size,
                    )

            optimize_t0 = time.perf_counter()
            training_infos = optimize_policy_once(
                policy=policy,
                preprocessor=preprocessors[primitive_id],
                optimizers=optimizers[primitive_id],
                online_iterator=online_iterators[primitive_id],
                offline_iterator=offline_iterators[primitive_id],
                optimization_step=optimization_steps[primitive_id],
                use_bc_update=is_dagger_bc_policy,
            )
            optimize_dt_s = time.perf_counter() - optimize_t0
            optimization_steps[primitive_id] += 1
            did_optimize = True

            training_infos["Optimization step"] = optimization_steps[primitive_id]
            training_infos["primitive_id"] = primitive_id
            training_infos["online_replay_buffer_size"] = len(replay_buffer)
            if offline_replay_buffer is not None:
                training_infos["offline_replay_buffer_size"] = len(offline_replay_buffer)
            training_infos["update_dt_ms"] = optimize_dt_s * 1000.0
            training_infos["update_frequency_hz"] = 1.0 / max(optimize_dt_s, 1e-9)

            if cfg.log_freq > 0 and optimization_steps[primitive_id] % cfg.log_freq == 0:
                loss_name, loss_value = primary_loss(training_infos, is_dagger_bc_policy=is_dagger_bc_policy)
                logging.info(
                    "[LEARNER] [%s] optimization_step=%s online=%s offline=%s update=%.1fHz %s=%.5f",
                    primitive_id,
                    optimization_steps[primitive_id],
                    len(replay_buffer),
                    len(offline_replay_buffer) if offline_replay_buffer is not None else 0,
                    training_infos["update_frequency_hz"],
                    loss_name,
                    loss_value,
                )

            if wandb_logger is not None:
                wandb_payload = {f"{primitive_id}/{k}": v for k, v in training_infos.items() if k != "primitive_id"}
                wandb_logger.log_dict(
                    d=wandb_payload,
                    mode="train",
                    custom_step_key=f"{primitive_id}/Optimization step",
                )

            if cfg.save_checkpoint and (
                optimization_steps[primitive_id] % cfg.save_freq == 0
                or optimization_steps[primitive_id] >= registry.online_step_budgets[primitive_id]
            ):
                save_training_checkpoint(
                    cfg=cfg,
                    primitive_id=primitive_id,
                    optimization_step=optimization_steps[primitive_id],
                    online_steps=registry.online_step_budgets[primitive_id],
                    interaction_message=last_interaction_messages.get(primitive_id),
                    policy=policy,
                    optimizers=optimizers[primitive_id],
                    replay_buffer=replay_buffer,
                    fps=cfg.env.fps,
                    preprocessor=preprocessors.get(primitive_id),
                    postprocessor=postprocessors.get(primitive_id),
                )

        if time.time() - last_push_t >= push_period_s:
            push_all_actor_policies_to_queue(parameters_queue, policies)
            last_push_t = time.time()

        if not did_optimize:
            time.sleep(0.01)

    shutdown_event.set()
    return {"optimization_steps": optimization_steps}


def optimize_policy_once(
    policy: SACPolicy,
    preprocessor: Any,
    optimizers: dict[str, Optimizer],
    online_iterator: Any | None,
    offline_iterator: Any | None,
    optimization_step: int,
    use_bc_update: bool = False,
) -> dict[str, float]:
    clip_grad_norm_value = policy.config.grad_clip_norm
    utd_ratio = max(1, int(policy.config.utd_ratio))
    policy_update_freq = max(1, int(policy.config.policy_update_freq))

    training_infos: dict[str, float] = {}
    if use_bc_update:
        if offline_iterator is None:
            raise RuntimeError("BC DAgger update requires an offline replay iterator.")
        batch = next(offline_iterator)
        forward_batch = prepare_forward_batch(policy=policy, preprocessor=preprocessor, batch=batch)
        if forward_batch is None:
            return training_infos
        bc_output = policy.forward(forward_batch, model="bc")
        loss_bc = bc_output["loss_bc"]
        optimizers["actor"].zero_grad()
        loss_bc.backward()
        bc_grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=policy.actor.parameters(),
            max_norm=clip_grad_norm_value,
        ).item()
        optimizers["actor"].step()
        training_infos.update(_floatify_training_infos(bc_output.get("training_infos", {})))
        training_infos["loss_bc"] = float(loss_bc.item())
        training_infos["bc_grad_norm"] = float(bc_grad_norm)
        return training_infos

    if online_iterator is None:
        raise RuntimeError("SAC update requires an online replay iterator.")

    for utd_step in range(utd_ratio):
        batch = next(online_iterator)
        if offline_iterator is not None:
            batch = concatenate_batch_transitions(
                left_batch_transitions=batch,
                right_batch_transition=next(offline_iterator),
            )
        forward_batch = prepare_forward_batch(policy=policy, preprocessor=preprocessor, batch=batch)
        if forward_batch is None:
            continue

        critic_output = policy.forward(forward_batch, model="critic")
        loss_critic = critic_output["loss_critic"]
        optimizers["critic"].zero_grad()
        loss_critic.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=policy.critic_ensemble.parameters(),
            max_norm=clip_grad_norm_value,
        ).item()
        optimizers["critic"].step()

        training_infos["loss_critic"] = float(loss_critic.item())
        training_infos["critic_grad_norm"] = float(critic_grad_norm)

        if policy.config.num_discrete_actions is not None:
            discrete_critic_output = policy.forward(forward_batch, model="discrete_critic")
            loss_discrete_critic = discrete_critic_output["loss_discrete_critic"]
            optimizers["discrete_critic"].zero_grad()
            loss_discrete_critic.backward()
            discrete_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters=policy.discrete_critic.parameters(),
                max_norm=clip_grad_norm_value,
            ).item()
            optimizers["discrete_critic"].step()
            training_infos["loss_discrete_critic"] = float(loss_discrete_critic.item())
            training_infos["discrete_critic_grad_norm"] = float(discrete_critic_grad_norm)

        if (optimization_step + utd_step) % policy_update_freq == 0:
            actor_output = policy.forward(forward_batch, model="actor")
            loss_actor = actor_output["loss_actor"]
            optimizers["actor"].zero_grad()
            loss_actor.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters=policy.actor.parameters(),
                max_norm=clip_grad_norm_value,
            ).item()
            optimizers["actor"].step()
            training_infos["loss_actor"] = float(loss_actor.item())
            training_infos["actor_grad_norm"] = float(actor_grad_norm)

            temperature_output = policy.forward(forward_batch, model="temperature")
            loss_temperature = temperature_output["loss_temperature"]
            optimizers["temperature"].zero_grad()
            loss_temperature.backward()
            temperature_grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters=[policy.log_alpha],
                max_norm=clip_grad_norm_value,
            ).item()
            optimizers["temperature"].step()
            training_infos["loss_temperature"] = float(loss_temperature.item())
            training_infos["temperature_grad_norm"] = float(temperature_grad_norm)
            training_infos["temperature"] = float(policy.temperature)

        policy.update_target_networks()

    return training_infos


def prepare_forward_batch(
    *,
    policy: SACPolicy,
    preprocessor: Any,
    batch: Any,
) -> dict[str, Any] | None:
    actions = batch[ACTION]
    rewards = batch["reward"]
    observations = batch["state"]
    next_observations = batch["next_state"]
    done = batch["done"]

    observations, actions, next_observations = preprocess_replay_batch(
        preprocessor=preprocessor,
        observations=observations,
        actions=actions,
        next_observations=next_observations,
    )

    if check_nan_in_transition(observations=observations, actions=actions, next_state=next_observations):
        return None

    observation_features, next_observation_features = get_observation_features(
        policy=policy,
        observations=observations,
        next_observations=next_observations,
    )

    return {
        ACTION: actions,
        "reward": rewards,
        "state": observations,
        "next_state": next_observations,
        "done": done,
        "observation_feature": observation_features,
        "next_observation_feature": next_observation_features,
        "complementary_info": batch.get("complementary_info"),
    }


def _floatify_training_infos(training_infos: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in training_infos.items():
        if isinstance(value, torch.Tensor):
            result[key] = float(value.item())
        elif isinstance(value, (int, float)):
            result[key] = float(value)
    return result


def make_optimizers(policy: SACPolicy) -> dict[str, Optimizer]:
    actor_lr = getattr(policy.config, "bc_lr", None)
    if actor_lr is None:
        actor_lr = policy.config.actor_lr
    optimizer_actor = torch.optim.Adam(
        params=[
            p
            for n, p in policy.actor.named_parameters()
            if not policy.config.shared_encoder or not n.startswith("encoder")
        ],
        lr=actor_lr,
    )
    optimizer_critic = torch.optim.Adam(params=policy.critic_ensemble.parameters(), lr=policy.config.critic_lr)
    optimizer_temperature = torch.optim.Adam(params=[policy.log_alpha], lr=policy.config.temperature_lr)
    optimizers: dict[str, Optimizer] = {
        "actor": optimizer_actor,
        "critic": optimizer_critic,
        "temperature": optimizer_temperature,
    }
    if policy.config.num_discrete_actions is not None:
        optimizers["discrete_critic"] = torch.optim.Adam(
            params=policy.discrete_critic.parameters(),
            lr=policy.config.critic_lr,
        )
    return optimizers


def push_all_actor_policies_to_queue(parameters_queue: Queue, policies: dict[str, SACPolicy]) -> None:
    def _drain_one(q: Queue) -> None:
        try:
            _ = q.get_nowait()
        except Exception:
            return

    for primitive_id, policy in policies.items():
        payload: dict[str, Any] = {
            "primitive_id": primitive_id,
            "policy": move_state_dict_to_device(policy.actor.state_dict(), device="cpu"),
        }
        if hasattr(policy, "discrete_critic") and policy.discrete_critic is not None:
            payload["discrete_critic"] = move_state_dict_to_device(
                policy.discrete_critic.state_dict(),
                device="cpu",
            )
        payload_bytes = state_to_bytes(payload)

        try:
            if parameters_queue.full():
                _drain_one(parameters_queue)
            parameters_queue.put(payload_bytes, block=False)
        except queue.Full:
            logging.warning("[LEARNER] parameters queue full, skipping push for primitive '%s'", primitive_id)


def process_transitions(
    transition_queue: Queue,
    replay_buffers: dict[str, ReplayBuffer],
    offline_replay_buffers: dict[str, ReplayBuffer | None],
    device: str,
    shutdown_event: Any,
) -> None:
    while not shutdown_event.is_set():
        try:
            packed_transitions = transition_queue.get_nowait()
        except queue.Empty:
            return

        transition_list = bytes_to_transitions(buffer=packed_transitions)
        for index, transition in enumerate(transition_list):
            primitive_id = transition.get("id")
            if primitive_id is None:
                primitive_id = transition.get("complementary_info", {}).get("primitive_id")
            if primitive_id is None or primitive_id not in replay_buffers:
                continue

            storage_device = replay_buffers[primitive_id].storage_device
            transition = move_transition_to_device(transition=transition, device=storage_device)
            if check_nan_in_transition(
                observations=transition["state"],
                actions=transition["action"],
                next_state=transition["next_state"],
            ):
                continue

            payload = {key: value for key, value in transition.items() if key != "id"}
            replay_buffers[primitive_id].add(**payload)

            offline_replay_buffer = offline_replay_buffers.get(primitive_id)
            if offline_replay_buffer is not None and transition.get("complementary_info", {}).get("is_intervention"):
                offline_replay_buffer.add(**payload)


def initialize_replay_buffers(
    *,
    cfg: MPNetTrainRLServerPipelineConfig,
    policies: dict[str, SACPolicy],
    device: str,
    storage_device: str,
) -> dict[str, ReplayBuffer]:
    replay_buffers: dict[str, ReplayBuffer] = {}
    for primitive_id, policy in policies.items():
        dataset_root = Path(cfg.output_dir) / primitive_id / "dataset"
        if cfg.resume and dataset_root.exists():
            repo_id = _dataset_repo_id(cfg=cfg, primitive_id=primitive_id)
            logging.info("[LEARNER] Loading online replay for primitive '%s' from %s", primitive_id, dataset_root)
            dataset = LeRobotDataset(repo_id=repo_id, root=str(dataset_root))
            replay_buffers[primitive_id] = ReplayBuffer.from_lerobot_dataset(
                lerobot_dataset=dataset,
                capacity=policy.config.online_buffer_capacity,
                device=device,
                storage_device=storage_device,
                state_keys=policy.config.input_features.keys(),
                optimize_memory=True,
            )
            continue

        replay_buffers[primitive_id] = ReplayBuffer(
            capacity=policy.config.online_buffer_capacity,
            device=device,
            storage_device=storage_device,
            state_keys=policy.config.input_features.keys(),
            optimize_memory=True,
        )

    return replay_buffers


def _extract_intervention_transitions(
    source_buffer: ReplayBuffer,
    dest_buffer: ReplayBuffer,
    max_count: int,
) -> int:
    """Copy is_intervention=True transitions from source into dest (newest first, up to max_count).

    source_buffer must have been loaded with optimize_memory=False so that next_states are
    materialised explicitly. Returns the number of transitions copied.
    """
    if len(source_buffer) == 0:
        return 0
    if not getattr(source_buffer, "has_complementary_info", False):
        return 0
    if "is_intervention" not in source_buffer.complementary_info:
        return 0

    size = len(source_buffer)
    is_intv = source_buffer.complementary_info["is_intervention"][:size].bool()
    copied = 0
    for logical_idx in range(size - 1, -1, -1):  # newest first
        if not bool(is_intv[logical_idx]):
            continue
        idx = (source_buffer.position - size + logical_idx) % source_buffer.capacity
        state = {k: source_buffer.states[k][idx].unsqueeze(0) for k in source_buffer.states}
        next_state = {k: source_buffer.next_states[k][idx].unsqueeze(0) for k in source_buffer.states}
        ci = (
            {k: source_buffer.complementary_info[k][idx].unsqueeze(0) for k in source_buffer.complementary_info_keys}
            if source_buffer.has_complementary_info
            else None
        )
        dest_buffer.add(
            state=state,
            action=source_buffer.actions[idx].unsqueeze(0),
            reward=float(source_buffer.rewards[idx]),
            next_state=next_state,
            done=bool(source_buffer.dones[idx]),
            truncated=bool(source_buffer.truncateds[idx]),
            complementary_info=ci,
        )
        copied += 1
        if copied >= max_count:
            break
    return copied


def initialize_offline_replay_buffers(
    *,
    cfg: MPNetTrainRLServerPipelineConfig,
    policies: dict[str, SACPolicy],
    device: str,
    storage_device: str,
) -> dict[str, ReplayBuffer | None]:
    offline_replay_buffers: dict[str, ReplayBuffer | None] = {}
    for primitive_id, policy in policies.items():
        capacity = policy.config.offline_buffer_capacity
        state_keys = policy.config.input_features.keys()

        offline_buffer = ReplayBuffer(
            capacity=capacity,
            device=device,
            storage_device=storage_device,
            state_keys=state_keys,
            optimize_memory=False,
        )

        # Phase A: resume — reconstruct intervention corrections from on-policy dataset
        if cfg.resume:
            legacy_offline_root = Path(cfg.output_dir) / primitive_id / "dataset-offline"
            if legacy_offline_root.exists():
                logging.warning(
                    "[LEARNER] Found legacy 'dataset-offline' folder at %s — it is no longer used. "
                    "The offline buffer is now reconstructed from the on-policy dataset. "
                    "You may delete this folder.",
                    legacy_offline_root,
                )

            online_dataset_root = Path(cfg.output_dir) / primitive_id / "dataset"
            if online_dataset_root.exists():
                repo_id = _dataset_repo_id(cfg=cfg, primitive_id=primitive_id)
                logging.info(
                    "[LEARNER] Reconstructing offline buffer for '%s' from on-policy dataset at %s",
                    primitive_id,
                    online_dataset_root,
                )
                online_dataset = LeRobotDataset(repo_id=repo_id, root=str(online_dataset_root))
                temp_buffer = ReplayBuffer.from_lerobot_dataset(
                    lerobot_dataset=online_dataset,
                    capacity=len(online_dataset),
                    device=storage_device,
                    storage_device=storage_device,
                    state_keys=state_keys,
                    optimize_memory=False,
                )
                n = _extract_intervention_transitions(temp_buffer, offline_buffer, capacity)
                logging.info(
                    "[LEARNER] Extracted %d intervention transitions into offline buffer for '%s'",
                    n,
                    primitive_id,
                )

        # Phase B: external offline demos (always attempted; fills remaining capacity)
        remaining = capacity - len(offline_buffer)
        if remaining > 0:
            external_root = _resolve_external_offline_dataset_root(cfg=cfg, primitive_id=primitive_id)
            if external_root is not None:
                repo_id = _dataset_repo_id(cfg=cfg, primitive_id=primitive_id)
                logging.info(
                    "[LEARNER] Loading external offline demos for '%s' from %s",
                    primitive_id,
                    external_root,
                )
                demo_dataset = LeRobotDataset(repo_id=repo_id, root=str(external_root))
                demo_buffer = ReplayBuffer.from_lerobot_dataset(
                    lerobot_dataset=demo_dataset,
                    capacity=min(len(demo_dataset), remaining),
                    device=device,
                    storage_device=storage_device,
                    state_keys=state_keys,
                    optimize_memory=False,
                )
                # Copy all demo transitions (no intervention filter needed for pre-collected demos)
                size = len(demo_buffer)
                for logical_idx in range(size - 1, -1, -1):
                    idx = (demo_buffer.position - size + logical_idx) % demo_buffer.capacity
                    state = {k: demo_buffer.states[k][idx].unsqueeze(0) for k in demo_buffer.states}
                    next_state = {k: demo_buffer.next_states[k][idx].unsqueeze(0) for k in demo_buffer.states}
                    ci = (
                        {k: demo_buffer.complementary_info[k][idx].unsqueeze(0) for k in demo_buffer.complementary_info_keys}
                        if demo_buffer.has_complementary_info
                        else None
                    )
                    offline_buffer.add(
                        state=state,
                        action=demo_buffer.actions[idx].unsqueeze(0),
                        reward=float(demo_buffer.rewards[idx]),
                        next_state=next_state,
                        done=bool(demo_buffer.dones[idx]),
                        truncated=bool(demo_buffer.truncateds[idx]),
                        complementary_info=ci,
                    )
                    if len(offline_buffer) >= capacity:
                        break

        offline_replay_buffers[primitive_id] = offline_buffer

    return offline_replay_buffers


def _resolve_external_offline_dataset_root(
    *,
    cfg: MPNetTrainRLServerPipelineConfig,
    primitive_id: str,
) -> Path | None:
    """Return the path to pre-collected external offline demos, if configured."""
    if cfg.dataset is None or cfg.dataset.root is None:
        return None
    dataset_root = Path(cfg.dataset.root)
    candidates = [
        dataset_root / primitive_id,
        dataset_root / "offline-demos" / primitive_id,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _dataset_repo_id(cfg: MPNetTrainRLServerPipelineConfig, primitive_id: str) -> str:
    if cfg.dataset is not None and cfg.dataset.repo_id:
        repo_id = cfg.dataset.repo_id
    else:
        repo_id = cfg.env.task or "mpnet"
    return f"{repo_id}-{primitive_id}"


def load_training_state(
    *,
    cfg: MPNetTrainRLServerPipelineConfig,
    optimizers: dict[str, dict[str, Optimizer]],
) -> tuple[dict[str, int] | None, dict[str, int] | None]:
    if not cfg.resume:
        return None, None

    optimization_steps: dict[str, int] = {}
    interaction_steps: dict[str, int] = {}
    try:
        for primitive_id, primitive_optimizers in optimizers.items():
            checkpoint_dir = Path(cfg.output_dir) / primitive_id / CHECKPOINTS_DIR / LAST_CHECKPOINT_LINK
            logging.info("[LEARNER] Loading %s training state from %s", primitive_id, checkpoint_dir)
            step, _, _ = utils_load_training_state(checkpoint_dir, primitive_optimizers, None)

            interaction_step = 0
            training_state_path = checkpoint_dir / TRAINING_STATE_DIR / "training_state.pt"
            if training_state_path.exists():
                training_state = torch.load(training_state_path, weights_only=False)
                interaction_step = int(training_state.get("interaction_step", 0))

            logging.info(
                "[LEARNER] Resuming %s from optimization step %s, interaction step %s",
                primitive_id,
                step,
                interaction_step,
            )
            optimization_steps[primitive_id] = int(step)
            interaction_steps[primitive_id] = interaction_step
    except Exception as exc:  # noqa: BLE001
        logging.error("[LEARNER] Failed to load training state: %s", exc)
        return None, None

    return optimization_steps, interaction_steps


def process_interaction_messages(
    interaction_message_queue: Queue,
    shutdown_event: Any,
    wandb_logger: WandBLogger | None,
    interaction_step_offset: dict[str, int],
    last_messages: dict[str, dict[str, Any]],
) -> None:
    while not shutdown_event.is_set():
        try:
            message = interaction_message_queue.get_nowait()
        except queue.Empty:
            return

        decoded = bytes_to_python_object(message)
        primitive_id = decoded.get("Primitive")
        if primitive_id is None:
            continue
        decoded["Interaction step"] = int(decoded.get("Interaction step", 0)) + interaction_step_offset.get(
            str(primitive_id),
            0,
        )
        last_messages[str(primitive_id)] = decoded

        if wandb_logger is not None:
            wandb_payload = {f"{primitive_id}/{key}": value for key, value in decoded.items() if key != "Primitive"}
            wandb_logger.log_dict(
                d=wandb_payload,
                mode="train",
                custom_step_key=f"{primitive_id}/Interaction step",
            )


def save_training_checkpoint(
    cfg: MPNetTrainRLServerPipelineConfig,
    primitive_id: str,
    optimization_step: int,
    online_steps: int,
    interaction_message: dict[str, Any] | None,
    policy: SACPolicy,
    optimizers: dict[str, Optimizer],
    replay_buffer: ReplayBuffer,
    fps: int,
    preprocessor: Any | None = None,
    postprocessor: Any | None = None,
) -> None:
    primitive_output_dir = Path(cfg.output_dir) / primitive_id
    checkpoint_dir = get_step_checkpoint_dir(primitive_output_dir, online_steps, optimization_step)
    try:
        with _checkpoint_safe_runtime_config(cfg):
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=optimization_step,
                cfg=cfg,
                policy=policy,
                optimizer=optimizers,
                scheduler=None,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
            )
    except Exception as exc:  # noqa: BLE001
        logging.warning(
            "[LEARNER] fallback checkpoint for primitive '%s' at step %s due to save_checkpoint error: %s",
            primitive_id,
            optimization_step,
            exc,
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": move_state_dict_to_device(policy.actor.state_dict(), device="cpu"),
                "critic": move_state_dict_to_device(policy.critic_ensemble.state_dict(), device="cpu"),
                "temperature": float(policy.temperature),
            },
            checkpoint_dir / "policy_fallback.pt",
        )
        torch.save(
            {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
            checkpoint_dir / "optimizers_fallback.pt",
        )

    training_state_dir = checkpoint_dir / TRAINING_STATE_DIR
    training_state_dir.mkdir(parents=True, exist_ok=True)
    interaction_step = int(interaction_message.get("Interaction step", 0)) if interaction_message is not None else 0
    torch.save(
        {"step": optimization_step, "interaction_step": interaction_step},
        training_state_dir / "training_state.pt",
    )

    update_last_checkpoint(checkpoint_dir)

    dataset_dir = primitive_output_dir / "dataset"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    repo_id = _dataset_repo_id(cfg=cfg, primitive_id=primitive_id)

    if len(replay_buffer) > 0:
        _save_replay_buffer_to_lerobot_dataset(
            replay_buffer,
            repo_id=repo_id,
            fps=fps,
            root=str(dataset_dir),
            task_name=primitive_id,
        )
    else:
        logging.info(
            "[LEARNER] Skipping online replay dataset save for primitive '%s' at step %s because the buffer is empty.",
            primitive_id,
            optimization_step,
        )



def _num_td_valid_offline_samples(replay_buffer: ReplayBuffer) -> int:
    if len(replay_buffer) == 0:
        return 0
    valid = _td_valid_offline_mask(replay_buffer)
    return int(valid.sum().item())


def _td_valid_offline_iterator(replay_buffer: ReplayBuffer, batch_size: int):
    while True:
        if _num_td_valid_offline_samples(replay_buffer) == 0:
            raise RuntimeError("Offline replay buffer has no TD-valid samples.")

        batch = None
        collected = 0
        while collected < batch_size:
            candidate = replay_buffer.sample(batch_size)
            keep = _td_valid_batch_mask(candidate)
            if not bool(keep.any()):
                continue
            candidate = _filter_batch_transition(candidate, keep)
            collected += int(keep.sum().item())
            if batch is None:
                batch = candidate
            else:
                batch = concatenate_batch_transitions(batch, candidate)

        if collected > batch_size:
            batch = _filter_batch_transition(
                batch,
                torch.arange(collected, device=batch[ACTION].device) < batch_size,
            )
        yield batch


def _td_valid_offline_mask(replay_buffer: ReplayBuffer) -> torch.Tensor:
    size = len(replay_buffer)
    return torch.ones(size, dtype=torch.bool, device=replay_buffer.storage_device)


def _td_valid_batch_mask(batch: dict[str, Any]) -> torch.Tensor:
    done = batch["done"].bool()
    return torch.ones_like(done, dtype=torch.bool)


def _filter_batch_transition(batch: dict[str, Any], keep: torch.Tensor) -> dict[str, Any]:
    keep = keep.to(device=batch[ACTION].device, dtype=torch.bool)
    filtered = dict(batch)
    filtered["state"] = {key: value[keep] for key, value in batch["state"].items()}
    filtered["next_state"] = {key: value[keep] for key, value in batch["next_state"].items()}
    for key in (ACTION, "reward", "done", "truncated"):
        filtered[key] = batch[key][keep]

    info = batch.get("complementary_info")
    if info is not None:
        filtered["complementary_info"] = {key: value[keep] for key, value in info.items()}
    return filtered


def _save_replay_buffer_to_lerobot_dataset(
    replay_buffer: ReplayBuffer,
    *,
    repo_id: str,
    fps: int,
    root: str,
    task_name: str = "from_replay_buffer",
):
    return _replay_buffer_to_lerobot_dataset(
        replay_buffer,
        repo_id=repo_id,
        fps=fps,
        root=root,
        task_name=task_name,
    )


@contextmanager
def _checkpoint_safe_runtime_config(cfg: MPNetTrainRLServerPipelineConfig):
    """Temporarily clear runtime-only config handles that checkpoint serialization cannot encode."""
    robot_cfgs = getattr(getattr(cfg, "env", None), "robot", None)
    if not isinstance(robot_cfgs, dict):
        yield
        return

    original_shm_managers: dict[str, Any] = {}
    try:
        for robot_name, robot_cfg in robot_cfgs.items():
            if robot_cfg is None or not hasattr(robot_cfg, "shm_manager"):
                continue
            original_shm_managers[robot_name] = robot_cfg.shm_manager
            robot_cfg.shm_manager = None
        yield
    finally:
        for robot_name, shm_manager in original_shm_managers.items():
            robot_cfg = robot_cfgs.get(robot_name)
            if robot_cfg is not None:
                robot_cfg.shm_manager = shm_manager


def _replay_buffer_to_lerobot_dataset(
    replay_buffer: ReplayBuffer,
    *,
    repo_id: str,
    fps: int,
    root: str,
    task_name: str,
) -> LeRobotDataset:
    if len(replay_buffer) == 0:
        raise ValueError("The replay buffer is empty. Cannot convert to a dataset.")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=root,
        robot_type=None,
        features=_replay_buffer_dataset_features(replay_buffer),
        use_videos=False,
    )

    pending_episode_frames = 0
    try:
        for idx in range(len(replay_buffer)):
            actual_idx = (replay_buffer.position - replay_buffer.size + idx) % replay_buffer.capacity
            frame_dict: dict[str, Any] = {
                ACTION: replay_buffer.actions[actual_idx].cpu(),
                "task": task_name,
            }

            for key in replay_buffer.states:
                frame_dict[key] = replay_buffer.states[key][actual_idx].cpu()

            frame_dict[REWARD] = torch.tensor([replay_buffer.rewards[actual_idx]], dtype=torch.float32).cpu()
            frame_dict[DONE] = torch.tensor([replay_buffer.dones[actual_idx]], dtype=torch.bool).cpu()

            if getattr(replay_buffer, "has_complementary_info", False):
                for key in replay_buffer.complementary_info_keys:
                    value = replay_buffer.complementary_info[key][actual_idx]
                    if isinstance(value, torch.Tensor) and value.ndim == 0:
                        value = value.unsqueeze(0)
                    frame_dict[f"complementary_info.{key}"] = value.cpu()

            dataset.add_frame(frame_dict)
            pending_episode_frames += 1
            if replay_buffer.dones[actual_idx] or replay_buffer.truncateds[actual_idx]:
                dataset.save_episode()
                pending_episode_frames = 0

        if pending_episode_frames > 0:
            dataset.save_episode()
    finally:
        if hasattr(dataset, "finalize"):
            dataset.finalize()

    return dataset


def _replay_buffer_dataset_features(replay_buffer: ReplayBuffer) -> dict[str, dict[str, Any]]:
    features = {
        "index": {"dtype": "int64", "shape": (1,), "names": None},
        "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
        "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
        "task_index": {"dtype": "int64", "shape": (1,), "names": None},
        ACTION: _tensor_to_dataset_feature(replay_buffer.actions[0]),
        REWARD: {"dtype": "float32", "shape": (1,), "names": None},
        DONE: {"dtype": "bool", "shape": (1,), "names": None},
    }

    for key, value in replay_buffer.states.items():
        features[key] = _tensor_to_dataset_feature(value[0])

    if getattr(replay_buffer, "has_complementary_info", False):
        for key in replay_buffer.complementary_info_keys:
            value = replay_buffer.complementary_info[key][0]
            if isinstance(value, torch.Tensor) and value.ndim == 0:
                value = value.unsqueeze(0)
            features[f"complementary_info.{key}"] = _tensor_to_dataset_feature(value)

    return features


def _tensor_to_dataset_feature(value: torch.Tensor) -> dict[str, Any]:
    if value.ndim == 0:
        value = value.unsqueeze(0)

    shape = tuple(value.shape)
    if len(shape) == 3 and shape[0] in (1, 3):
        return {
            "dtype": "image",
            "shape": shape,
            "names": ["channels", "height", "width"],
        }

    if value.dtype == torch.bool:
        dtype = "bool"
    elif value.dtype.is_floating_point:
        dtype = "float32"
    elif value.dtype in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        dtype = "int64"
    else:
        raise TypeError(f"Unsupported replay buffer tensor dtype '{value.dtype}' for dataset export.")

    return {
        "dtype": dtype,
        "shape": shape,
        "names": None,
    }


def start_learner_server(
    registry: Any,
    shutdown_event: Any,
    parameters_queue: Queue,
    transition_queue: Queue,
    interaction_message_queue: Queue,
) -> None:
    transport_cfg = registry.actor_learner_policy_cfg.actor_learner_config
    service = LearnerService(
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        seconds_between_pushes=transport_cfg.policy_parameters_push_frequency,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_message_queue,
        queue_get_timeout=transport_cfg.queue_get_timeout,
    )

    server = grpc.server(
        ThreadPoolExecutor(max_workers=MAX_WORKERS),
        options=[
            ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
            ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
        ],
    )
    services_pb2_grpc.add_LearnerServiceServicer_to_server(service, server)
    server.add_insecure_port(f"{transport_cfg.learner_host}:{transport_cfg.learner_port}")
    server.start()
    logging.info(
        "[LEARNER] gRPC server started at %s:%s",
        transport_cfg.learner_host,
        transport_cfg.learner_port,
    )

    shutdown_event.wait()
    server.stop(SHUTDOWN_TIMEOUT)


def _all_finished(steps: dict[str, int], budgets: dict[str, int]) -> bool:
    return all(steps[primitive_id] >= budgets[primitive_id] for primitive_id in budgets)


def _use_threads(policy_cfg: SACPolicy | Any) -> bool:
    return policy_cfg.concurrency.learner == "threads"


if __name__ == "__main__":
    import experiments
    train_cli()
