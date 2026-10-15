from dataclasses import dataclass

from beast_refiner.async_inference import AsyncInferenceConfig

from share.configs.record import RecordConfig


@dataclass
class RecordBeastConfig(RecordConfig):
    # Run beast_xvla_refiner_v2 policies with System 2 (VLM + coarse plan, ~40 ms) on a background thread and
    # System 1 (spline refiner, ~6 ms) on every control tick, so the control loop never waits for a new plan.
    # None keeps the synchronous `policy.select_action` path. Enable it with e.g.
    # `--async_inference.sys2_trigger_margin_ticks=4` (any AsyncInferenceConfig field can be set this way).
    # Needs policies with inference_mode="both" and use_amp=False, running at the fps they were trained at.
    # Lives here, not in RecordConfig, so the plain recording config does not depend on beast_refiner.
    async_inference: AsyncInferenceConfig | None = None
