from dataclasses import dataclass

from share.configs.mpnet import DatasetRecordConfig, TrainRLServerPipelineConfig
from share.configs.rl import MPNetTrainRLServerPipelineConfig
from share.debug.mpnet_debug import MPNetDebugConfig


@dataclass(kw_only=True)
class RecordConfig(MPNetTrainRLServerPipelineConfig):
    debug: MPNetDebugConfig | None = None
    # Whether record should load and run the configured policy.
    use_policy: bool = True
    # Whether to save only intervention/correction steps instead of the full rollout.
    save_only_interventions: bool = False
    # Display all cameras on screen
    display_data: bool = False
    # Display data on a remote Rerun server
    display_ip: str | None = None
    # Port of the remote Rerun server
    display_port: int | None = None
    # Whether to  display compressed images in Rerun
    display_compressed_images: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Stop the whole run once the graph reaches a terminal primitive, instead of looping
    # back into a fresh episode. Used by the teach env, whose graph ends in a terminal
    # StoreTaughtPose primitive, so teaching one pose is a single self-terminating run.
    stop_after_terminal: bool = False
    # Ornstein-Uhlenbeck exploration noise added to the policy's action during autonomous
    # (non-intervention) steps -- see share.utils.exploration. 0 disables it. Raise if
    # rollouts still succeed too reliably to teach a critic anything (aim for roughly
    # 30-65% success); lower if the robot moves unsafely. Everything else about the noise
    # (per-dimension scale, correlation time) is derived automatically.
    exploration_noise_scale: float = 0.0
    # Seconds before the noise decorrelates; a reaction-time constant, not a training
    # hyperparameter, so this default is rarely worth changing.
    exploration_noise_correlation_s: float = 0.5
