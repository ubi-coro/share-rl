# Episode-save benchmark

Goal: shorten the pause at the end of every episode in `record_beast.py` (`dataset.save_episode()`, reported as
30 s+ for the 4-camera hoodie-folding setup), without changing what ends up in the dataset.

`LOG.md` is the experiment log (what was tried, measured, kept or dropped). This file explains the method.

## What is measured

`bench_episode_save.py` replays a recorded dataset through the exact calls `record_loop` makes, with no robot:

* per step: build the observation like `ObservationProcessor._process_image` does (float32 CHW images in
  [0, 1], strided view from HWC), then `dataset.add_frame(frame)`;
* 30 Hz pacing (`--pace realtime`, default) so background work (PNG writers, streaming encoders) overlaps the
  episode the way it does on the robot. `--pace fast` is only for smoke tests: it backs up the PNG writer and
  inflates the save time;
* at the end of the episode, `dataset.save_episode()` is timed. That is the number we want to reduce.

Reported per run (also written to `results/<label>-<timestamp>.json`):

| metric | meaning |
| --- | --- |
| `save_s_mean_steady` | headline: mean `save_episode()` time over episodes 1..N-1 |
| `save_s_first` | episode 0 (includes one-off setup: first video file, `info.json` video info, ...) |
| `add_frame_ms_p99_worst` / `max_worst` | cost of `add_frame` on the control thread; the whole loop budget is 33 ms |
| `loop_overruns_total` | steps where obs build + `add_frame` alone exceeded 1/fps |
| `finalize_s` | `dataset.finalize()` at shutdown |
| `dataset_size_mb` | output size |
| `verify.*` | (`--verify`) output re-decoded and compared to the source: frame counts, state/action, image PSNR |

`--profile` adds a per-phase breakdown of the save (`other/encode` is the residual: for the PNG path this is the
video encode running in a process pool, which cannot be timed from the parent).

## Setup used

* Source: `HoodieFolding__Task_InitialAdversarial__Lights_Off__191225` (19 episodes, ~1600 frames = ~53 s,
  4 cameras 640x480 AV1, 14-dim state/action), copied from `/vol` (ceph) to local disk once.
* Episodes 0-4 are replayed (the 200 MB video-file roll-over is not reached; later episodes would pay slightly
  more for the video concatenation).
* Machine: 32 cores, 125 GB RAM, RTX 4090. `record_beast` writes to `HF_LEROBOT_HOME`, which is on
  `/media/internal/hdd` (a **rotational** disk, ext4 over LUKS/LVM). `/vol/coro` is ceph. The local NVMe is 100 % full
  and cannot be used.
* Baseline create kwargs are what `make_policies_and_datasets` passes today: `libsvtav1`, `batch_encoding_size=1`,
  16 image-writer threads (4 per camera), no streaming encoding.

## Running

```bash
cd src/share/scripts/benchmarks/episode_save
python bench_episode_save.py --label baseline-hdd --root <dir on the disk under test> --profile --verify
# any LeRobotDataset.create kwarg can be overridden, e.g.
python bench_episode_save.py --label streaming --root <dir> --create-kw streaming_encoding=true
```

## Caveats

* The replay does not run the policy or the camera threads, so CPU contention on the robot PC is lower than in
  a real run. The benchmark therefore reports `add_frame` latency and overruns, but a real recording with a
  policy in the loop is still the final check for anything that moves work into the episode.
* Numbers come from one machine and a handful of episodes; differences below ~10 % are noise.
