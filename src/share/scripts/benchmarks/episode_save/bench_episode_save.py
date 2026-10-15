"""Benchmark how long `dataset.save_episode()` takes in the record_beast.py data path.

A recorded LeRobot dataset is replayed through the same `add_frame()` / `save_episode()` calls that
`record_loop` makes (float32 CHW image tensors, 30 Hz pacing), with no robot attached. Per episode we
record the save time, the cost of every `add_frame()` (this runs on the control thread, so it must stay
far below the 33 ms frame budget) and, with `--profile`, a per-phase breakdown of the save.

Example:
    python bench_episode_save.py --label baseline --root /media/internal/hdd/jstranghoener/tmp/bench
    python bench_episode_save.py --label streaming --create-kw streaming_encoding=true

See README.md / LOG.md in this directory for the experiment log.
"""

import argparse
import json
import os
import shutil
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.robot_utils import precise_sleep

from share.utils.video_utils import use_zero_copy_streaming_encoder

DEFAULT_SOURCE = "/media/internal/hdd/jstranghoener/tmp/claude-30410/-home-jstranghoener-PycharmProjects-share-rl/63bd66ae-f399-4b97-b9bc-e495d7734b9f/scratchpad/source/adversarial_191225"
RESULTS_DIR = Path(__file__).parent / "results"
DERIVED_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


# --------------------------------------------------------------------------- source dataset ----------


def _decode_segment(video_path: Path, from_timestamp: float, length: int, fps: int) -> np.ndarray:
    """Decode `length` frames starting at `from_timestamp` of a (multi-episode) video file as uint8 HWC."""
    start = round(from_timestamp * fps)
    out = None
    n = 0
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if start > 0:
            container.seek(int(max(start - 2, 0) / fps / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            idx = round(float(frame.pts * stream.time_base) * fps)
            if idx < start:
                continue
            if idx >= start + length:
                break
            if out is None:
                out = np.empty((length, frame.height, frame.width, 3), dtype=np.uint8)
            out[idx - start] = frame.to_ndarray(format="rgb24")
            n += 1
    if n != length:
        raise RuntimeError(f"decoded {n} frames from {video_path}, expected {length}")
    return out


def load_episodes(root: Path, episode_indices: list[int], max_frames: int | None = None) -> tuple[dict, list[dict]]:
    """Load episodes of a LeRobot v3 dataset into memory: images as uint8 HWC, the rest as numpy."""
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = info["fps"]
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    meta = pd.concat([pd.read_parquet(f) for f in sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))])
    data = pd.concat([pd.read_parquet(f) for f in sorted((root / "data").glob("chunk-*/file-*.parquet"))])

    episodes = []
    with ThreadPoolExecutor(max_workers=len(video_keys)) as pool:
        for ep in episode_indices:
            row = meta[meta["episode_index"] == ep].iloc[0]
            length = int(row["length"])
            if max_frames is not None:
                length = min(length, max_frames)
            df = data[data["episode_index"] == ep].sort_values("frame_index").iloc[:length]
            futures = {}
            for key in video_keys:
                path = root / info["video_path"].format(
                    video_key=key,
                    chunk_index=int(row[f"videos/{key}/chunk_index"]),
                    file_index=int(row[f"videos/{key}/file_index"]),
                )
                futures[key] = pool.submit(
                    _decode_segment, path, float(row[f"videos/{key}/from_timestamp"]), length, fps
                )
            episodes.append(
                {
                    "episode_index": ep,
                    "length": length,
                    "task": str(row["tasks"][0]),
                    "images": {k: f.result() for k, f in futures.items()},
                    "vectors": {
                        k: np.stack(df[k].to_numpy()).astype(np.float32).reshape(length, -1)
                        for k, v in info["features"].items()
                        if v["dtype"] in ("float32", "bool") and k not in DERIVED_FEATURES
                    },
                }
            )
    return info, episodes


# --------------------------------------------------------------------------- replay ----------


def make_frame(episode: dict, t: int, features: dict) -> dict:
    """Build one `record_loop`-style frame: float32 CHW image tensors in [0, 1] (see `_process_image`)."""
    frame = {}
    for key, imgs in episode["images"].items():
        img = torch.from_numpy(imgs[t]).permute(2, 0, 1)  # einops "h w c -> c h w": a strided view
        img = img.to(torch.float32)
        frame[key] = img / 255.0 if img.max() > 1.0 else img
    for key, vec in episode["vectors"].items():
        dtype = np.bool_ if features[key]["dtype"] == "bool" else np.float32
        value = vec[t].astype(dtype)
        frame[key] = value if key.startswith("next.") else torch.from_numpy(value)
    frame["task"] = episode["task"]
    return frame


class PhaseTimer:
    """Accumulates wall time of selected lerobot internals during `save_episode`."""

    def __init__(self):
        self.totals: dict[str, float] = defaultdict(float)
        self.dropped_frames = 0

    def wrap(self, owner, attr: str, label: str):
        orig = getattr(owner, attr, None)
        if orig is None:
            return

        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return orig(*args, **kwargs)
            finally:
                self.totals[label] += time.perf_counter() - t0

        setattr(owner, attr, timed)

    def install(self):
        from lerobot.datasets import dataset_metadata, dataset_writer, video_utils

        W = dataset_writer.DatasetWriter
        self.wrap(W, "_wait_image_writer", "wait_image_writer")
        self.wrap(dataset_writer, "compute_episode_stats", "episode_stats")
        self.wrap(W, "_save_episode_data", "parquet_data")
        self.wrap(W, "_save_episode_video", "save_episode_video")
        self.wrap(dataset_writer, "concatenate_video_files", "  concat_videos (in save_episode_video)")
        self.wrap(dataset_metadata.LeRobotDatasetMetadata, "save_episode", "meta_save_episode")
        self.wrap(video_utils.StreamingVideoEncoder, "finish_episode", "streaming_finish_episode")

    def count_dropped_frames(self):
        """A dropped frame silently desyncs video and parquet, so count them for every run."""
        from lerobot.datasets.video_utils import StreamingVideoEncoder as encoder_cls

        orig = encoder_cls.finish_episode

        def finish(encoder):
            self.dropped_frames += sum(encoder._dropped_frames.values())
            return orig(encoder)

        encoder_cls.finish_episode = finish

    def pop(self) -> dict[str, float]:
        out = dict(self.totals)
        self.totals.clear()
        return out


def cpu_seconds() -> float:
    """User + system CPU seconds of this process, its threads and its (waited-for) children."""
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


def run(args) -> dict:
    source_root = Path(args.source)
    t0 = time.perf_counter()
    info, episodes = load_episodes(source_root, args.episodes, args.max_frames)
    print(f"loaded {len(episodes)} episodes from {source_root} in {time.perf_counter() - t0:.1f}s", flush=True)

    fps = info["fps"]
    features = {k: dict(v) for k, v in info["features"].items() if k not in DERIVED_FEATURES}
    for ft in features.values():
        ft.pop("info", None)
        ft["shape"] = tuple(ft["shape"])  # env_to_dataset_features passes tuples, JSON gives lists

    run_root = Path(args.root) / f"{args.label}-{datetime.now():%Y%m%d-%H%M%S}"
    create_kwargs = dict(
        image_writer_processes=0,
        image_writer_threads=4 * len(episodes[0]["images"]),  # num_image_writer_threads_per_camera=4 default
        batch_encoding_size=1,
        vcodec="libsvtav1",
    )
    for item in args.create_kw:
        k, v = item.split("=", 1)
        create_kwargs[k] = json.loads(v) if v[:1] in "[{0123456789-" or v in ("true", "false", "null") else v
    print("create kwargs:", create_kwargs, "| root:", run_root, flush=True)

    dataset = LeRobotDataset.create(
        "bench/replay", fps, root=run_root, features=features, robot_type=info.get("robot_type"), **create_kwargs
    )
    if args.zero_copy:
        use_zero_copy_streaming_encoder(dataset)
    if args.meta_kw:
        dataset.meta.update_chunk_settings(**{k: json.loads(v) for k, v in (i.split("=", 1) for i in args.meta_kw)})
    phases = PhaseTimer()
    phases.count_dropped_frames()
    if args.profile:
        phases.install()

    results = []
    for i, episode in enumerate(episodes):
        add_ms, obs_ms = [], []
        overruns = 0
        dropped_before = phases.dropped_frames
        t_episode = time.perf_counter()
        cpu0 = cpu_seconds()
        for t in range(episode["length"]):
            t_loop = time.perf_counter()
            frame = make_frame(episode, t, features)
            t_add = time.perf_counter()
            dataset.add_frame(frame)
            now = time.perf_counter()
            obs_ms.append((t_add - t_loop) * 1e3)
            add_ms.append((now - t_add) * 1e3)
            if args.pace == "realtime":
                if now - t_loop > 1 / fps:
                    overruns += 1
                precise_sleep(1 / fps - (time.perf_counter() - t_loop))
        t_replayed = time.perf_counter()
        cpu1 = cpu_seconds()

        phases.pop()
        t_save0 = time.perf_counter()
        dataset.save_episode()
        save_s = time.perf_counter() - t_save0
        cpu2 = cpu_seconds()
        breakdown = phases.pop()

        add = np.asarray(add_ms)
        res = {
            "episode": episode["episode_index"],
            "frames": episode["length"],
            "save_s": save_s,
            "replay_s": t_replayed - t_episode,
            "cpu_cores_during_episode": (cpu1 - cpu0) / (t_replayed - t_episode),
            "cpu_s_in_save": cpu2 - cpu1,
            "add_frame_ms": {
                "mean": float(add.mean()),
                "p50": float(np.percentile(add, 50)),
                "p99": float(np.percentile(add, 99)),
                "max": float(add.max()),
            },
            "obs_build_ms_mean": float(np.mean(obs_ms)),
            "loop_overruns": overruns,
            "dropped_frames": phases.dropped_frames - dropped_before,
            "phases_s": breakdown,
        }
        results.append(res)
        extra = ""
        if breakdown:
            top = ("wait_image_writer", "episode_stats", "parquet_data", "save_episode_video",
                   "meta_save_episode", "streaming_finish_episode")
            other = save_s - sum(breakdown.get(k, 0.0) for k in top)
            extra = " | " + ", ".join(f"{k.strip()}={v:.1f}" for k, v in breakdown.items()) + f", other/encode={other:.1f}"
        print(
            f"[ep {i}] save={save_s:6.2f}s  add_frame mean/p99/max={res['add_frame_ms']['mean']:.2f}/"
            f"{res['add_frame_ms']['p99']:.2f}/{res['add_frame_ms']['max']:.1f}ms  obs_build={res['obs_build_ms_mean']:.1f}ms  "
            f"overruns={overruns}  dropped={res['dropped_frames']}  "
            f"cpu_cores(episode)={res['cpu_cores_during_episode']:.1f}  cpu_s(save)={res['cpu_s_in_save']:.0f}{extra}",
            flush=True,
        )

    t_fin = time.perf_counter()
    dataset.finalize()
    finalize_s = time.perf_counter() - t_fin

    size_mb = sum(f.stat().st_size for f in run_root.rglob("*") if f.is_file()) / 1e6
    saves = [r["save_s"] for r in results]
    steady = saves[1:] or saves
    summary = {
        "label": args.label,
        "create_kwargs": create_kwargs,
        "root": str(run_root),
        "pace": args.pace,
        "episodes": [r["episode"] for r in results],
        "frames_per_episode": [r["frames"] for r in results],
        "save_s_all": saves,
        "save_s_mean": statistics.mean(saves),
        "save_s_mean_steady": statistics.mean(steady),
        "save_s_first": saves[0],
        "save_s_max": max(saves),
        "add_frame_ms_p99_worst": max(r["add_frame_ms"]["p99"] for r in results),
        "add_frame_ms_max_worst": max(r["add_frame_ms"]["max"] for r in results),
        "loop_overruns_total": sum(r["loop_overruns"] for r in results),
        "dropped_frames_total": phases.dropped_frames,
        "cpu_cores_during_episode_mean": statistics.mean(r["cpu_cores_during_episode"] for r in results),
        "cpu_s_in_save_mean": statistics.mean(r["cpu_s_in_save"] for r in results),
        "finalize_s": finalize_s,
        "dataset_size_mb": size_mb,
    }

    if args.verify:
        summary["verify"] = verify(run_root, source_root, args.episodes, args.max_frames)

    print("\n== SUMMARY ==")
    for k, v in summary.items():
        if k != "save_s_all":
            print(f"  {k}: {v}")
    print("  save_s per episode:", [round(s, 2) for s in saves])

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{args.label}-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"summary": summary, "episodes": results}, indent=2, default=str))
    print("results:", out)

    if not args.keep:
        shutil.rmtree(run_root, ignore_errors=True)
    return summary


# --------------------------------------------------------------------------- verification ----------


def verify(out_root: Path, source_root: Path, episodes: list[int], max_frames: int | None) -> dict:
    """Re-open the written dataset and compare it to the source: frame counts, vectors, image PSNR."""
    n = len(episodes)
    _, written = load_episodes(out_root, list(range(n)), max_frames)
    _, source = load_episodes(source_root, episodes, max_frames)
    psnrs, vec_ok = [], True
    for w, s in zip(written, source, strict=True):
        assert w["length"] == s["length"], (w["length"], s["length"])
        for key, vec in s["vectors"].items():
            vec_ok &= bool(np.allclose(vec, w["vectors"][key]))
        for key, imgs in s["images"].items():
            idx = np.linspace(0, s["length"] - 1, 20).round().astype(int)
            a = imgs[idx].astype(np.float32)
            b = w["images"][key][idx].astype(np.float32)
            mse = ((a - b) ** 2).mean(axis=(1, 2, 3))
            psnrs.append(float((10 * np.log10(255.0**2 / np.maximum(mse, 1e-9))).mean()))
    return {"frames_match": True, "vectors_match": vec_ok, "image_psnr_db_mean": float(np.mean(psnrs))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", required=True)
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--root", required=True, help="parent dir for the benchmark dataset (the disk under test)")
    p.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--max-frames", type=int, default=None, help="truncate episodes (quick smoke tests)")
    p.add_argument("--pace", choices=["realtime", "fast"], default="realtime")
    p.add_argument("--create-kw", nargs="*", default=[], help="extra LeRobotDataset.create kwargs, key=json")
    p.add_argument("--meta-kw", nargs="*", default=[],
                   help="dataset.meta.update_chunk_settings kwargs, e.g. video_files_size_in_mb=1")
    p.add_argument("--torch-threads", type=int, default=None,
                   help="torch.set_num_threads; 1 keeps torch's OpenMP pool from masking the encoders' CPU use")
    p.add_argument("--zero-copy", action="store_true", help="use share's ZeroCopyStreamingVideoEncoder")
    p.add_argument("--profile", action="store_true", help="per-phase breakdown of save_episode")
    p.add_argument("--verify", action="store_true", help="re-open output and compare against the source")
    p.add_argument("--keep", action="store_true", help="keep the written dataset")
    args = p.parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    run(args)


if __name__ == "__main__":
    main()
