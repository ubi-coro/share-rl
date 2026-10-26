import logging
import sys
import time
from contextlib import contextmanager
from copy import copy
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence, Tuple

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor.rename_processor import rename_stats
from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile

from share.configs.record import RecordConfig
from share.envs.manipulation_primitive.config_manipulation_primitive import ManipulationPrimitiveConfig
from share.envs.manipulation_primitive_net.transitions import OnTimeLimit
from share.envs.utils import env_to_dataset_features, resolve_value, to_scalar
from share.utils.video_utils import use_zero_copy_streaming_encoder

if TYPE_CHECKING:
    from beast_refiner.async_inference import AsyncInferenceConfig, AsyncPolicyRunner
    from lerobot.policies.pretrained import PreTrainedPolicy


def make_policies_and_datasets(cfg: RecordConfig):
    datasets = {}
    policies = {}
    preprocessors = {}
    postprocessors = {}
    rename_map = cfg.rename_map
    for name, p in cfg.env.primitives.items():
        if p.is_adaptive:

            if name == cfg.env.reset_primitive:
                continue

            # 1) dataset
            stats = None
            if cfg.dataset is not None:
                root = Path(cfg.dataset.root) / name
                repo_id = f"{cfg.dataset.repo_id}-{name}"

                # Streaming encoding feeds frames straight to the video encoders, so no image writer is needed.
                streaming = cfg.dataset.streaming_encoding
                writer_kwargs = dict(
                    batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                    vcodec=cfg.dataset.vcodec,
                    image_writer_processes=0 if streaming else cfg.dataset.num_image_writer_processes,
                    image_writer_threads=0
                    if streaming
                    else cfg.dataset.num_image_writer_threads_per_camera * p.num_cameras,
                    streaming_encoding=streaming,
                    encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                    encoder_threads=cfg.dataset.encoder_threads,
                )

                if cfg.resume:
                    datasets[name] = LeRobotDataset.resume(repo_id, root=root, **writer_kwargs)

                else:
                    datasets[name] = LeRobotDataset.create(
                        repo_id,
                        cfg.env.fps,
                        root=root,
                        features=env_to_dataset_features(p.features),
                        robot_type=cfg.env.type,
                        use_videos=cfg.dataset.video,
                        **writer_kwargs,
                    )

                if streaming:
                    use_zero_copy_streaming_encoder(datasets[name])

                if cfg.dataset.video_files_size_in_mb is not None:
                    datasets[name].meta.update_chunk_settings(
                        video_files_size_in_mb=cfg.dataset.video_files_size_in_mb
                    )

                stats = rename_stats(datasets[name].meta.stats, rename_map)

            # 2) policy
            if p.policy is None:
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

            policies[name] = make_policy(cfg=p.policy, env_cfg=p, rename_map=rename_map)
            policies[name] = policies[name].eval()

            pre, post = make_pre_post_processors(
                policy_cfg=p.policy,
                pretrained_path=str(p.policy.pretrained_path),
                dataset_stats=stats,
                preprocessor_overrides={
                    "device_processor": {"device": p.policy.device},
                    "rename_observations_processor": {"rename_map": rename_map},
                },
            )
            preprocessors[name] = pre
            postprocessors[name] = post

    return datasets, policies, preprocessors, postprocessors


def predict_action_async(
    observation: dict[str, Any],
    runner: "AsyncPolicyRunner",
    obs_timestep: int,
    device: torch.device,
    preprocessor,
    postprocessor,
    task: str | None = None,
    robot_type: str | None = None,
) -> torch.Tensor:
    """`lerobot.utils.control_utils.predict_action` with the policy call routed through an `AsyncPolicyRunner`.

    Same observation preparation, pre- and post-processing; only `policy.select_action(batch)` is replaced by
    `runner.select_action(batch, obs_timestep)`, which answers from System 1 without waiting for System 2.
    `obs_timestep` has to increase by one per control step and restart at 0 every episode.

    Runs under `no_grad`, not `inference_mode` like the synchronous path: the batch is handed to the runner's
    System 2 thread, and inference tensors cannot be updated in place outside of inference mode. Autocast is
    thread-local and not applied either, so the policy must have `use_amp=False`.
    """
    observation = copy(observation)
    with torch.no_grad():
        observation = prepare_observation_for_inference(observation, device, task, robot_type)
        observation = preprocessor(observation)
        action = runner.select_action(observation, obs_timestep)
        return postprocessor(action)


def make_async_runners(
    policies: dict[str, "PreTrainedPolicy | None"],
    async_cfg: "AsyncInferenceConfig | None",
    fps: float,
) -> dict[str, "AsyncPolicyRunner"]:
    """Started `AsyncPolicyRunner`s for the beast_xvla_refiner_v2 policies, if `async_cfg` is set.

    That includes the policies built on its model (e.g. beast_xvla_refiner_v2_ladder). Primitives whose policy
    is something else (or has none) are left out and keep the synchronous `policy.select_action` path. Call after every policy is loaded: `apply_process_tuning` freezes the GC
    for the objects alive at that point. `fps` is the rate the control loop runs at.
    """
    if async_cfg is None:
        return {}

    # only needed, and slow to import, when async inference is actually requested
    from beast_refiner.async_inference import AsyncPolicyRunner, apply_process_tuning
    from beast_refiner.policies.beast_xvla_refiner_v2.modeling_beast_xvla_refiner_v2 import BeastXVLARefinerV2Model

    runners: dict[str, AsyncPolicyRunner] = {}
    for name, policy in policies.items():
        if policy is None:
            continue
        if not isinstance(getattr(policy, "model", None), BeastXVLARefinerV2Model):
            logging.warning(
                f"[{name}] async inference only supports beast_xvla_refiner_v2, "
                f"running {policy.config.type} synchronously."
            )
            continue
        if policy.config.use_amp:
            raise ValueError(
                f"[{name}] async inference does not apply autocast on its System 2 thread: set use_amp=False "
                f"(policy_overwrites={{'use_amp': False}})."
            )
        if policy.model.fps != fps:
            logging.warning(
                f"[{name}] the policy was trained at {policy.model.fps} fps but the env runs at {fps} fps; "
                f"System 1's velocity pin and the replan margin assume the training rate."
            )
        # refuses inference_mode="sys2": System 1 has to run every tick
        runners[name] = AsyncPolicyRunner(policy, async_cfg)

    if runners:
        apply_process_tuning(async_cfg)
    started = []
    try:
        for name, runner in runners.items():
            logging.info(f"[{name}] starting async inference (warming up System 1 and System 2)")
            started.append(runner)
            runner.start(tune_process=False)
    except Exception:
        for runner in started:
            runner.stop()
        raise
    return runners


class TickStats:
    """Aggregates the control loop's tick times so they can be logged once per window, off the hot path.

    A synchronous write to a slow stderr (pipe, tmux) blocks the control loop, which is what a per-tick INFO
    line did to the tick after it.
    """

    def __init__(self, window: int):
        self.window = max(1, window)
        self.dt_loop: list[float] = []
        self.dt_load: list[float] = []
        self.phases: dict[str, list[float]] = {}

    def add(self, dt_loop: float, dt_load: float, **phases: float) -> str | None:
        """Record one tick; returns the summary of the window when it is full, else None.

        `phases` are named durations (seconds) of parts of the tick, reported as mean/max in ms. A phase that
        only occurs on some ticks (e.g. no policy on a teleoperated primitive) is averaged over those.
        """
        self.dt_loop.append(dt_loop)
        self.dt_load.append(dt_load)
        for name, duration in phases.items():
            self.phases.setdefault(name, []).append(duration)
        if len(self.dt_loop) < self.window:
            return None
        summary = (
            f"last {len(self.dt_loop)} ticks: "
            f"dt_loop mean {1000 * sum(self.dt_loop) / len(self.dt_loop):5.2f}ms "
            f"max {1000 * max(self.dt_loop):5.2f}ms ({1 / max(self.dt_loop):3.1f}hz min), "
            f"dt_load max {1000 * max(self.dt_load):5.2f}ms"
        )
        if self.phases:
            summary += " | mean/max ms: " + ", ".join(
                f"{name} {1000 * sum(values) / len(values):.1f}/{1000 * max(values):.1f}"
                for name, values in self.phases.items()
            )
        self.dt_loop.clear()
        self.dt_load.clear()
        self.phases.clear()
        return summary


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


def mute_log_messages(*prefixes: str) -> None:
    """Drop log records whose message starts with one of `prefixes` from the root handlers.

    Filters by message because some libraries log on the root logger, so their level cannot be set per module.
    Call after `init_logging()`, which replaces the root handlers.
    """
    for handler in logging.getLogger().handlers:
        handler.addFilter(lambda record: not record.getMessage().startswith(prefixes))


@contextmanager
def _console_logs_via_tqdm() -> Iterator[None]:
    """Route the root console handlers through `tqdm.write` so log lines do not tear a progress bar.

    Unlike `tqdm.contrib.logging.logging_redirect_tqdm` this keeps the handlers themselves, so their level and
    filters (see `mute_log_messages`) stay in effect.
    """
    handlers = [h for h in logging.getLogger().handlers if getattr(h, "stream", None) in (sys.stdout, sys.stderr)]
    streams = [handler.setStream(DummyTqdmFile(handler.stream)) for handler in handlers]  # returns the old stream
    try:
        yield
    finally:
        for handler, stream in zip(handlers, streams):
            handler.setStream(stream)


@contextmanager
def episode_progress(mp_net, dataset: LeRobotDataset | None = None) -> Iterator[Callable[[dict], None] | None]:
    """Console progress line for the episode about to be recorded; yields a per-step `on_step(info)` callback.

    Shows the episode number (saved episodes + 1) if the active primitive records into `dataset`, and a bar of
    elapsed seconds against the primitive's `OnTimeLimit` if it has one, e.g. `grab ep 7  40%|####| 10/25s`.
    Yields None, and shows nothing, if there is neither. The line is removed when the episode ends and is
    disabled when stderr is not a terminal.
    """
    primitive = mp_net.active_primitive
    fps = mp_net.config.fps
    limit = next(
        (t for t in mp_net.config.transitions if t.source == primitive and isinstance(t, OnTimeLimit)), None
    )
    if limit is None and dataset is None:
        yield None
        return

    label = primitive if dataset is None else f"{primitive} ep {dataset.num_episodes + 1}"
    step_key = "primitive_step" if limit is None else limit.step_key
    bar_format = "{desc} {n:.0f}s" if limit is None else "{desc} {percentage:3.0f}%|{bar:20}| {n:.0f}/{total:.0f}s"

    def on_step(info: dict) -> None:
        try:
            seconds = to_scalar(resolve_value(info, step_key)) / fps
        except KeyError:  # e.g. an intervention-completed step carries no step counters
            return
        bar.update(seconds - bar.n)

    with _console_logs_via_tqdm(), tqdm(
        total=None if limit is None else limit.max_steps / fps,
        desc=label, bar_format=bar_format, mininterval=0.5, leave=False, disable=None,
    ) as bar:
        yield on_step


def confirm(question: str) -> bool:
    """Ask a y/n question on the console until it is answered. A closed stdin counts as no."""
    with tqdm.external_write_mode():  # clears a progress bar, so the prompt gets a line of its own
        while True:
            try:
                answer = input(f"{question} [y/n] ").strip().lower()
            except EOFError:
                return False
            if answer in ("y", "n"):
                return answer == "y"


def offer_to_save_episode(dataset: LeRobotDataset | None) -> None:
    """After a failed step, ask whether to keep the frames recorded so far and save them if so."""
    frames = 0 if dataset is None else dataset.writer.episode_buffer["size"]
    if frames > 0 and confirm(f"Save the {frames} frames recorded so far as episode {dataset.num_episodes}?"):
        dataset.save_episode()
        logging.info("Saved the partial episode.")

