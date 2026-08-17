import logging
import time
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence, Tuple

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.processor.rename_processor import RenameObservationsProcessorStep, rename_stats
from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME

from share.configs.record import RecordConfig
from share.rl.runtime import resolve_policy_dataset_stats
from share.envs.manipulation_primitive.config_manipulation_primitive import ManipulationPrimitiveConfig
from share.envs.utils import env_to_dataset_features

try:
    import batch_rl.policies  # noqa: F401  (registers ditflow/chunk_critic policy types)
except ImportError:
    pass


_OFFLINE_VISION_CACHE_PREFIX = "observation.cache."
_LIVE_IMAGE_PREFIX = "observation.images."


def _remove_redundant_offline_cache_features(policy_cfg, env_features: dict) -> None:
    """Drop saved vision-cache inputs when the equivalent live image is available."""
    input_features = policy_cfg.input_features or {}
    env_keys = set(env_features)
    removable = []
    unavailable = []
    for key in input_features:
        if not key.startswith(_OFFLINE_VISION_CACHE_PREFIX) or key in env_keys:
            continue
        camera_name = key.removeprefix(_OFFLINE_VISION_CACHE_PREFIX)
        image_key = f"{_LIVE_IMAGE_PREFIX}{camera_name}"
        if image_key in input_features and image_key in env_keys:
            removable.append(key)
        else:
            unavailable.append(key)
    if unavailable:
        raise ValueError(
            "Policy requires offline-only vision cache features without equivalent "
            f"live images: {unavailable}"
        )
    if removable:
        policy_cfg.input_features = {
            key: feature for key, feature in input_features.items() if key not in removable
        }
        logging.info("Ignoring offline-only policy cache features at deployment: %s", removable)


def make_policies_and_datasets(cfg: RecordConfig):
    datasets = {}
    policies = {}
    preprocessors = {}
    postprocessors = {}
    for name, p in cfg.env.primitives.items():
        if p.is_adaptive:

            if name == cfg.env.reset_primitive:
                continue

            # 1) dataset
            rename_map = {}
            stats = None
            if cfg.dataset is not None and p.policy is not None:
                root = Path(cfg.dataset.root) / name
                repo_id = f"{cfg.dataset.repo_id}-{name}"

                if cfg.resume:
                    datasets[name] = LeRobotDataset.resume(
                        repo_id,
                        root=root,
                        batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                        vcodec=cfg.dataset.vcodec,
                        image_writer_processes=cfg.dataset.num_image_writer_processes,
                        image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * p.num_cameras,
                    )
                    if "rl.is_intervention" not in datasets[name].features:
                        logging.warning(
                            f"Resumed dataset '{repo_id}' was recorded before the 'rl.is_intervention' "
                            f"feature was added; adding frames with this feature will fail. "
                            f"Record into a fresh dataset instead."
                        )

                else:
                    datasets[name] = LeRobotDataset.create(
                        repo_id,
                        cfg.env.fps,
                        root=root,
                        features=env_to_dataset_features(p.features),
                        robot_type=cfg.env.type,
                        use_videos=cfg.dataset.video,
                        image_writer_processes=cfg.dataset.num_image_writer_processes,
                        image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * p.num_cameras,
                        batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                        vcodec=cfg.dataset.vcodec,
                    )

                rename_map = cfg.dataset.rename_map
                stats = rename_stats(datasets[name].meta.stats, rename_map)

            # 2) policy
            if not cfg.use_policy or p.policy is None:
                policies[name] = None
                preprocessors[name] = None
                postprocessors[name] = None
                continue

            policy_path = p.policy.pretrained_path
            if policy_path is None:
                assert cfg.dataset is not None, "Policies that are not loaded from checkpoints need a dataset"
            else:
                p.policy = PreTrainedConfig.from_pretrained(p.policy.pretrained_path)
                p.policy = replace(p.policy, **p.policy_overwrites)
                p.policy.pretrained_path = policy_path
                _remove_redundant_offline_cache_features(p.policy, p.features or {})

            policies[name] = make_policy(cfg=p.policy, env_cfg=p)
            policies[name] = policies[name].eval()

            # Checkpoints saved by our learner do not include processor pipelines, so
            # rebuild them from the policy config the same way training did instead of
            # loading them from the checkpoint.
            processor_path = p.policy.pretrained_path
            if processor_path is not None and not (
                Path(processor_path) / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
            ).exists():
                processor_path = None
                config_stats = resolve_policy_dataset_stats(p.policy)
                if config_stats is not None:
                    stats = config_stats

            pre, post = make_pre_post_processors(
                policy_cfg=p.policy,
                pretrained_path=processor_path,
                dataset_stats=stats,
                preprocessor_overrides={
                    "device_processor": {"device": p.policy.device},
                    "rename_observations_processor": {"rename_map": rename_map},
                },
            )
            if processor_path is None:
                # Freshly built pipelines ignore `preprocessor_overrides`, so apply the
                # record-time rename map to the fresh preprocessor directly.
                for step in pre.steps:
                    if isinstance(step, RenameObservationsProcessorStep):
                        step.rename_map = rename_map
            preprocessors[name] = pre
            postprocessors[name] = post

    return datasets, policies, preprocessors, postprocessors


def predict_action(
    *,
    observation: dict[str, Any],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
) -> torch.Tensor:
    """Run policy inference from Share's processed tensor observation contract."""

    policy_observation = {}
    for key in policy.config.input_features:
        value = observation[key]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        if value.ndim == len(policy.config.input_features[key].shape):
            value = value.unsqueeze(0)
        policy_observation[key] = value.to(device)

    policy_observation["task"] = task if task else ""
    policy_observation["robot_type"] = robot_type if robot_type else ""

    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        policy_observation = preprocessor(policy_observation)
        action = policy.select_action(policy_observation)
        action = postprocessor(action)

    return action


def make_step_timing_hooks(
    pipeline_steps: Sequence["ProcessorStep"],
    label: str = "pipeline",
    log_every: int = 1,
    ema_alpha: float = 0.2,
    also_print: bool = False,
) -> Tuple:
    """
    Create before/after hooks that time each step in a DataProcessorPipeline.

    Args:
        pipeline_steps: the pipeline steps whose steps we are timing.
        label: a short label to identify this pipeline in logs (e.g., "env", "action").
        log_every: emit a summary every N pipeline passes.
        ema_alpha: smoothing factor for EMA timings (0..1].
        also_print: if True, print the summary in addition to logging.

    Returns:
        before_hook, after_hook callables suitable for pipeline.before_step_hooks / after_step_hooks.
    """
    step_names: Sequence[str] = [type(s).__name__ for s in pipeline_steps]
    n_steps = len(step_names)

    # Per-step timing state
    t_start = [0.0] * n_steps              # last start time per step
    last_ms = [0.0] * n_steps              # last measured dt per step (ms)
    ema_ms  = [0.0] * n_steps              # EMA per step (ms)

    # Per-pass timing
    pass_idx = 0
    pass_t0 = 0.0

    def _emit():
        # Compose a compact, single-line breakdown
        parts = [f"{step_names[i]}={last_ms[i]:.2f}ms(ema:{ema_ms[i]:.2f})"
                 for i in range(n_steps)]
        total_last = sum(last_ms)
        total_ema  = sum(ema_ms)
        msg = f"[{label}] total={total_last:.2f}ms(ema:{total_ema:.2f}) | " + ", ".join(parts)
        logging.info(msg)
        if also_print:
            print(msg)

    def before_hook(idx: int, _transition: "EnvTransition") -> None:
        nonlocal pass_t0
        # If first step, mark pipeline-pass start
        if idx == 0:
            pass_t0 = time.perf_counter()
        t_start[idx] = time.perf_counter()

    def after_hook(idx: int, _transition: "EnvTransition") -> None:
        nonlocal pass_idx
        dt_ms = (time.perf_counter() - t_start[idx]) * 1000.0
        last_ms[idx] = dt_ms
        # Update EMA
        ema_ms[idx] = dt_ms if ema_ms[idx] == 0.0 else (1.0 - ema_alpha) * ema_ms[idx] + ema_alpha * dt_ms

        # If last step, bump pass counter and (maybe) emit
        if idx == n_steps - 1:
            pass_idx += 1
            if log_every > 0 and (pass_idx % log_every == 0):
                # Optionally include the pipeline wall time (may differ slightly from sum of steps)
                pipe_ms = (time.perf_counter() - pass_t0) * 1000.0
                # Replace total with measured wall time if you prefer:
                # msg can include pipe_ms too; here we just keep it implicit to keep line short.
                _emit()

    return [before_hook], [after_hook]


class MPNetStepCounter:
    def __init__(self, primitives: dict[str, ManipulationPrimitiveConfig]):
        # initialize per-primitive step budgets and counters
        self._budget: dict[str, int] = {}
        self._count: dict[str, int] = {}
        self._last_finish_count: dict[str, int] = {}
        for name, p in primitives.items():
            self._last_finish_count[name] = 0
            self._count[name] = 0

    def __getitem__(self, item):
        return self._count[item]

    def increment(self, name: str, n: int = 1):
        """Call this every time the given primitive takes n interaction steps."""
        if name in self._count:
            self._count[name] += n

    def finish_episode(self, name: str):
        """True if this primitive is non-adaptive or has reached its online_steps."""
        if name in self._count:
            self._last_finish_count[name] = self._count[name]

    def episode_length(self, name: str) -> int:
        return self._count.get(name, 0) - self._last_finish_count.get(name, 0)

    @property
    def global_step(self):
        return sum(self._count.values())


@contextmanager
def suppress_logging(level=logging.CRITICAL):
    previous = logging.root.manager.disable
    logging.disable(level)
    try:
        yield
    finally:
        logging.disable(previous)
