# Experiment log: shortening `save_episode()` in `record_beast.py`

Method, metrics and setup are in `README.md`. All runs: episodes 0-4 of the replayed dataset (~1600 frames,
4 cameras), 30 Hz pacing, output on the local HDD unless the row says otherwise. "Steady" = mean over
episodes 1-4. Raw per-episode numbers are in `results/*.json`.

## Summary table

| # | variant | save (steady) | `add_frame` mean / p99 / max | CPU cores during episode | verdict |
| --- | --- | --- | --- | --- | --- |
| 0 | baseline (PNG + post-hoc AV1, HDD) | **43.8 s** | 0.24 / 0.5-1.2 / 10 ms | n/a | reference |
| 1 | `streaming_encoding=True` (lerobot built-in) | 1.3 s | 10.4 / 20-25 / 77 ms | n/a | save fixed, but 10 ms on the control thread: not usable as is |
| 2 | 1 + zero-copy `feed_frame` | **1.1 s** | 0.13 / 0.18 / 1 ms (one 111 ms outlier) | 12.6 (see 3) | works |
| 3 | 2 + `encoder_queue_maxsize=150`, `encoder_threads` 1/2/4 | 1.15 s (2 threads) | 0.12-0.14 / 0.18 / 1.3 ms | ~+0.8 over PNG baseline | adopted on paper, **reverted after 6** |
| 6 | 3 on real hardware, live control loop | n/a | n/a | n/a | **REGRESSION: 30 -> 15 Hz on the real robot. `streaming_encoding` defaults flipped back to `False` in `BEASTDatasetConfig` / `AlohaSingleDatasetConfig`.** |

(rows are appended below as experiments finish)

## 0. Baseline

What `make_policies_and_datasets` does today: `LeRobotDataset.create(..., vcodec="libsvtav1",
batch_encoding_size=1, image_writer_threads=16)`. Frames are written as PNGs by 16 threads while recording;
`save_episode()` then (a) loads ~300 sampled PNGs per camera back from disk for the image statistics and
(b) decodes all PNGs again and encodes them to AV1 in a 4-process pool, (c) concatenates onto the running
video file, (d) writes parquet + metadata.

| episode | save | stats from PNGs | encode (residual) | parquet | concat |
| --- | --- | --- | --- | --- | --- |
| 0 | 43.7 s | 15.5 | 27.6 | 0.6 | - |
| 1 | 44.7 s | 15.1 | 28.6 | 0.6 | 0.2 |
| 2 | 46.0 s | 16.6 | 28.4 | 0.6 | 0.4 |
| 3 | 40.3 s | 13.8 | 25.5 | 0.6 | 0.5 |
| 4 | 44.3 s | 17.7 | 25.4 | 0.6 | 0.6 |

Mean 43.8 s (reported on the robot: "30 s+"; the replay is slightly slower because the HDD is also serving other
users). Output 641 MB for 5 episodes, image PSNR vs. source 40.8 dB (AV1 crf 30, same for every variant).

Take-aways: 98 % of the time is the two image phases, both caused by the PNG round-trip (disk write, disk read
for stats, PNG decode, then encode). Everything else (parquet, metadata, concat) is ~1 s together.

## 1. Built-in streaming encoder (`streaming_encoding=True`, `image_writer_threads=0`)

Frames are handed to one encoder thread per camera while recording; no PNGs exist. Image statistics are computed
incrementally in the encoder threads.

* save 0.7 / 1.1 / 1.6 / 1.2 / 1.5 s (rises with the video file being concatenated onto, see 4)
* dropped frames: 0, output identical in size and PSNR (40.82 dB) to the baseline.
* **But** `add_frame` costs 10.4 ms on average (p99 20-25 ms, max 77 ms) and 1381 of 8122 steps overran
  33 ms (baseline: 216; its `obs_build` time was not logged in that run, see 0b). Cause: upstream `feed_frame` does
  `image.copy()` of four float32 640x480x3 arrays (15 MB) on the calling thread.

## 2. Zero-copy `feed_frame` (`share.utils.video_utils.ZeroCopyStreamingVideoEncoder`)

Skip the copy; the encoder thread reads the tensor the loop handed over. Safety argument: the PNG
`AsyncImageWriter` that the baseline uses keeps the same un-copied references (`image.cpu().numpy()` is a
view for CPU tensors) until a worker writes them, so the baseline already relies on frames not being mutated
after `add_frame`. The class docstring states the assumption.

* save 0.70 / 0.96 / 1.10 / 1.16 / 1.30 s, mean steady 1.13 s (**-97 %**)
* `add_frame` 0.13 ms mean, p99 0.18 ms; overruns 9 of 8122; dropped 0
* the benchmark's own `obs_build` also fell from 10-26 ms to 1.8 ms: the copies had been thrashing memory
  bandwidth for the whole process.
* one `add_frame` of 111 ms in episode 2 (the 0.1 s `put` timeout on a full queue: with the default
  `queue_maxsize=30` a short encoder stall fills the queue). Larger queue in the next runs.
* encoders use **12.6 CPU cores** on average during the episode (SVT-AV1 `lp` defaults to all threads, per
  camera). On the benchmark box that is free; next to the policy, camera and teleop threads it is a risk
  for control-loop timing. -> experiment 3.

## 3. Queue depth and encoder threads

Zero-copy encoder, `encoder_queue_maxsize=150` (5 s of frames; the 111 ms outlier of run 2 did not recur),
`encoder_threads` = SVT-AV1 `lp` per camera. Full-length episodes, HDD:

| `encoder_threads` | save per episode (s) | steady mean | dropped | `add_frame` p99 |
| --- | --- | --- | --- | --- |
| unset (all cores), queue 30 | 0.70 0.96 1.10 1.16 1.30 | 1.13 | 0 | 0.18 ms |
| 2 | 0.72 0.98 1.12 1.18 1.34 | 1.15 | 0 | 0.18 ms |
| 4 | 0.69 1.06 1.07 2.90 1.30 | 1.58 | 0 | 0.18 ms |
| 1 | 0.81 1.23 1.25 2.53 1.42 | 1.61 | 0 | 0.17 ms |

The 2.5-2.9 s episodes (both at episode 3) are I/O noise on the shared HDD, not the encoder (episode 3 is 1.2 s
in the other runs). Output size and PSNR are identical in all runs (641 MB, 40.82 dB).

**CPU attribution.** The "12.6 cores" of run 2 were mostly the benchmark's own `torch` ops (float conversion of
four images per step runs on torch's OpenMP pool) and did not change with `encoder_threads`, so that number says
nothing about the encoders. Re-measured with `--torch-threads 1`, 3 episodes of 600 frames (20 s), HDD:

| variant | cores busy during the episode | CPU-s inside `save_episode` | save |
| --- | --- | --- | --- |
| baseline (PNG writers) | 3.0 | 104 | 15.8 s |
| streaming, all threads | 4.0 | 1.9 | 0.48 s |
| streaming, 4 threads | 4.0 | 1.8 | 0.43 s |
| streaming, **2 threads** | 3.8 | 1.5 | 0.50 s |
| streaming, 1 thread | 3.2 | 1.2 | 0.90 s |

Compared with the PNG baseline, moving the encode into the episode costs about **+0.8 busy cores** with 2
threads (+1.0 unrestricted, +0.15 with 1 thread), while the total CPU per episode roughly halves: the baseline
pays ~100 CPU-s again inside `save_episode`. (The split of the 3.0 / 3.8 cores between control thread and
background work was not measured separately; the replay's `precise_sleep` also spins for its last 10 ms.) With 1 thread per camera the encoder
runs close to real time and the backlog doubles the save time, so there is no headroom; **2 threads** is the
setting used in `BEASTDatasetConfig`. It is a trade-off the robot PC has to confirm: with the policy and the
camera threads competing for cores, `encoder_threads=1` is the fallback if control-loop timing suffers.

## 4. Where is the dataset written? (HDD vs tmpfs vs `/vol` ceph)

Same replay, output root on different storage (4-5 episodes, `--profile`). `/media/internal/nvme` is full and was
not usable. In the harness shell `TMPDIR` pointed at the HDD, which matters for the concat column (see below).

| output root | baseline save (steady) | stats from PNGs | concat | streaming + zero-copy save (steady) | concat |
| --- | --- | --- | --- | --- | --- |
| local HDD (where `record_beast` writes) | 43.8 s | 15.8 s | 0.4 s | 1.15 s | 0.4 s |
| tmpfs `/dev/shm` (RAM, upper bound) | 41.0 s | 14.1 s | 2.4 s | 3.31 s | 2.5 s |
| `/vol/coro` (ceph) | 43.7 s | 14.9 s | 2.4 s | 3.43 s | 2.5 s |

* **The baseline is CPU-bound, not disk-bound**: even in RAM it needs 41 s (about 280 CPU-s per save: decode 4x1600
  PNGs, encode AV1, stats). Putting the dataset on a faster disk or avoiding `/vol` saves at most ~3 s of 44. Copying
  recordings to local disk is still sensible for other reasons, but it is not the fix.
* The streaming path removes the CPU work, so what is left is I/O-shaped. The apparent HDD-vs-tmpfs/ceph gap
  (1.15 s vs 3.3-3.4 s) is **not** about those disks being slower: tmpfs and ceph give the same 2.5 s concat,
  and the only difference to the HDD runs is that lerobot's `concatenate_video_files` writes its temp output to
  `tempfile.gettempdir()` and then `shutil.move`s it into the dataset. With `TMPDIR` on the HDD and the dataset on the
  HDD that move is a rename; with the dataset elsewhere it is a full copy of the growing video files (4 cameras).
  The user's own shells have `TMPDIR` unset (so `/tmp` on the system NVMe), i.e. the real setup always pays the
  copy. Experiment 5 re-measures with that.
* Concat grows with every episode appended to the same video file (1.7 -> 2.5 -> 3.2 s on tmpfs/ceph) because each
  save remuxes the whole file (up to `video_files_size_in_mb=200`) per camera.

## 5. Video concatenation / file layout

Streaming + zero-copy, 2 encoder threads, queue 150, `TMPDIR` unset (as in the user's shells, so lerobot's
`concatenate_video_files` temp file lands on the system NVMe, not next to the dataset).

**With normal ~200 MB video files**, HDD, 7 full-length episodes (one interrupted run, no `--verify`, so treat
as indicative only): save 0.72 1.10 1.30 1.42 3.05 1.77 0.99 s, concat 0.4 0.5 0.7 2.3 1.1 0.3 s. Episode 6 starts
a new video file (200 MB limit reached) and drops back to no concat cost. Within a file concat still grows
(0.4 -> 0.7 s before the 2.3 s outlier at episode 4, unexplained); this is *not* consistently slower than the
`TMPDIR`-on-HDD runs in section 4 (1.15 s steady), so the "temp-file copy" explanation there is a hypothesis, not
a confirmed cause.

**One file per episode** (`meta.update_chunk_settings(video_files_size_in_mb=1)`, so `concatenate_video_files` is
never called), 5 full-length episodes, `--verify`:

| storage | save per episode (s) | steady mean | `add_frame` p99 | dropped | verify |
| --- | --- | --- | --- | --- | --- |
| HDD (5 episodes) | 0.80 0.82 0.74 0.70 0.74 | **0.75** | 0.18 ms | 0 | frames/vectors match, PSNR 40.82 dB |
| `/vol` ceph (3 episodes) | 0.74 0.79 0.91 | 0.85 | 0.18 ms | 0 | frames/vectors match, PSNR 40.86 dB |

Both are flat across episodes (no growth, no 2-3 s spikes) and verified bit-identical in content to every other
variant; `/vol` runs ~0.1 s higher than HDD, consistent with the small gap seen for every variant in section 4.
**This is the layout to adopt**: it removes the concat step entirely instead of papering over it, at the cost of
many small files per episode instead of ~200 MB files (a different on-disk layout from existing datasets;
LeRobot v3 reads per-episode metadata regardless of file size, so loading is unaffected). `BEASTDatasetConfig`
should set `dataset.meta.update_chunk_settings(video_files_size_in_mb=1)` (or an equivalent small value) after
creation, alongside `streaming_encoding=True`.

## 6. Real-hardware regression: streaming encoders starve camera read threads (2026-10-08)

`streaming_encoding=True` (experiments 1-3) was adopted and shipped as the default in `BEASTDatasetConfig` and
`AlohaSingleDatasetConfig`. On the actual robot PC, with a real `aloha_single` teleop recording running
(`manual_reset` primitive, no policy), the control loop dropped from the configured 30 Hz to a steady ~15 Hz
(`dt_loop` == `dt_load`, 63-68 ms per step -- confirmed via the loop's own `dt_loop`/`dt_load` log line, not a
`precise_sleep` miscalculation: all the time is in the per-step work, none of it in sleeping).

Root cause: lerobot's `Camera.async_read()` does not do blocking hardware I/O itself -- it waits (up to 200 ms)
on an event set by the camera's own dedicated background capture thread, then returns the latest buffered
frame. `ViperX.get_observation()` calls `async_read()` for all 4 cameras sequentially in a plain Python loop
(`src/share/robots/viperx/lerobot_robot_viperx/viperx.py:231-235`). In steady state this should be near-instant
for every camera whose background thread is keeping up; if it instead takes ~67 ms total, the background
capture threads themselves are not delivering at the configured fps.

This is exactly the gap flagged in the README caveats but never actually tested: every benchmark run in
experiments 0-5 replays frames from RAM directly into `add_frame()` -- there are no real camera background
threads, no real robot serial I/O, and no real competing hardware-driven threads to starve. `streaming_encoding`
moves AV1 encoding (SVT-AV1's own native thread pool, `encoder_threads=2` per camera x 4 cameras) from
*after* the episode into *during* it, now running concurrently with the cameras' and robot's own background
threads for the first time. The live process showed 141 OS threads and ~3.8-4 busy cores (`/proc/<pid>/task/*`)
while idle-looking in aggregate `ps`/`top` snapshots -- consistent with GIL/scheduling contention starving the
camera read threads specifically, rather than raw CPU saturation.

Confirmed causal, not coincidental: the user reports this control-loop rate was correct before these changes
shipped, with no other change to the hardware or wiring in between.

**Action taken:** `streaming_encoding` defaults flipped back to `False` in both dataset configs (keeps
`video_files_size_in_mb=1`, which only affects `save_episode()`'s post-episode concat and has no live-loop
cost). This restores the original ~44 s save stall, but a stalled save is a nuisance; a halved control-loop
rate silently corrupts every episode's timing.

**Not yet done / open for next time:**
- A clean, isolated A/B confirmation (`--dataset.streaming_encoding=false` vs `=true` on an otherwise identical
  live run) was not actually run -- the fix here is based on the user's report that 15 Hz did not occur before
  these changes, which is strong but not a controlled test on this exact host under this exact load.
- If streaming encoding is revisited: benchmark it with the *real* cameras and robot running (not a RAM
  replay), e.g. by timing `get_observation()` directly while the streaming encoders are active, before
  trusting any synthetic number again. Candidates to try: `encoder_threads=1` (lighter per-camera footprint,
  still verified correct in experiment 3), or pinning the encoder threads to specific cores (`taskset`/cgroup)
  away from the cameras' and robot's own threads.

## 7. Root cause confirmed, and a real fix (2026-10-08, same day as experiment 6)

Per-camera timing was added temporarily to `ManipulationPrimitive._get_observation()`
(`src/share/envs/manipulation_primitive/env_manipulation_primitive.py`) -- NOT
`ViperX.get_observation()`, which has its own camera-timing debug log but an empty `self.cameras`
dict in this architecture (cameras are connected centrally by `ManipulationPrimitiveNet` and read in
`ManipulationPrimitive`, not per-robot); that log line never fires and is a dead end for this issue.

Result: every step, `cam_low` alone took 55-60 ms; `cam_top` and both RealSense wrist cameras were
~0 ms. This isolates the entire 15 Hz regression to one specific camera, and rules out experiment 6's
`streaming_encoding` hypothesis completely (confirmed independently: disabling it changed nothing).

Two further physical tests on `cam_low` also came back negative, ruling out two more hypotheses:
* Replugging the USB cable -- ruled out a stuck UVC negotiation state.
* Connecting it directly to the workstation instead of through its USB hub -- ruled out hub
  bandwidth contention (its hub also carried both right-arm Dynamixel serial adapters, which was a
  plausible, topology-backed theory, but the direct connection showed the exact same 55-60 ms).

**Actual root cause: lighting.** `cam_low` is the Logitech C922 Pro Stream Webcam pointed at a dimmer
part of the scene; shining a light on it immediately fixed the frame rate. The C922 is documented to
auto-throttle its own frame rate under auto-exposure in low light to maintain image quality -- a
firmware-level behavior, invisible to and unfixable from any OS/driver/USB-level setting, and
orthogonal to every change made in this session (dataset config, streaming encoding, USB topology).

**Fix implemented** (not a workaround for lighting specifically -- a correctness fix for the control
loop that helps with *any* camera that can't keep up, for any reason): `_get_observation()` now peeks
`Camera.read_latest(max_age_ms=ManipulationPrimitive.CAMERA_MAX_AGE_MS)` instead of blocking on
`Camera.async_read()`. A camera that's keeping up still gets a fresh frame every step; a camera that's
lagging (low light, momentary USB hiccup, anything) now gets its last buffered frame reused instead of
stalling every other camera and the whole control loop, as long as that frame is within
`CAMERA_MAX_AGE_MS` (100 ms, ~3 frames at 30 fps). A camera that is genuinely dead/disconnected still
raises instead of silently feeding a frame from minutes ago. This matters beyond today's bug: this lab
runs deliberate low-light ("Lights_Off") experiments elsewhere, where forcing fixed/manual exposure to
prevent this would conflict with the experiment itself -- the peek-based read handles that case too,
gracefully, without needing to touch exposure settings at all.

Covered by `tests/share/envs/test_manipulation_primitive_observation.py`: a keeping-up camera is
unaffected, a lagging-but-within-bound camera is tolerated, a camera past the bound still raises.

**Still true from experiment 6, independent of this fix:** `streaming_encoding` defaults stay `False`
in both dataset configs. It was not the cause of this specific bug, but the thread-starvation question
it raised (141 OS threads, ~3.8-4 busy cores live, vs. a synthetic benchmark that never exercised real
camera/robot background threads) was never actually isolated with a controlled A/B test. That remains
open, separate, lower-priority follow-up -- not resolved by anything in this section.
