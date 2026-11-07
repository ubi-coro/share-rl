"""Run requested autonomous rollouts of LeRobot SAC/BC policies on an MP-Net, one seeded episode at a time.

A request is a JSON file in `<queue>/requests/` (written by an outside program, e.g. a weight-editing harness):

    {"id": "pass_03",
     "out": "/abs/path/to/pass_03",                      # where episodes are written
     "primitive": "insert",                             # the adaptive primitive the policy drives
     "policies": {"original": "/abs/.../pretrained_model", "edited": "/abs/.../pretrained_model"},
     "episodes": [{"key": "edited_0007", "policy": "edited", "seed": 7}, ...]}

Requests run in file-name order, episodes in the listed order. Each episode starts with `mp_net.reset(seed=seed)`, so
the same seed gives the same start state (for envs whose reset primitives draw from the env RNG), and the policy acts
with its deterministic action: tanh of the actor's mean, unnormalized. When an episode ends it is written atomically:

    <out>/<key>.npz          what the policy saw and did in the adaptive primitive, per step:
                             state (T, S) raw, image (T, 3, h, w) uint8 crop, fmap (T, C, H, W) float16 frozen-encoder
                             features computed live from the exact policy input, action (T, A); plus success, steps,
                             duration_s, reason
    <out>/full/<key>.npz     the uncropped camera frames of the same steps, JPEG-encoded
    <out>/hidden/<key>.json  diagnostics that are not policy inputs: the absolute task-frame TCP pose per step

When every episode of a request exists, `<queue>/done/<id>.json` is written. An episode that is interrupted (crash,
network error, operator restart) leaves no file and is simply run again with the same seed: `--supervise` restarts the
runner after any failure, and nothing in the episode files records that a restart happened. Restarts are logged to
`<queue>/runner.log` for the operator only.

    python -m share.scripts.rollout_runner --queue=/path/queue --env_module=wsp.real.han_insertion \
        --env_type=ur3e_han_insertion --supervise
    # headless dry run on the toy rig: add --mock=wsp.real.han_insertion_mock --no_realtime
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

POLL_S = 2.0
RESTART_DELAY_S = 5.0


def _log(queue: Path, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(queue / "runner.log", "a") as f:
        f.write(line + "\n")


def _atomic_savez(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


class DeterministicPolicy:
    """A LeRobot SAC-family policy acting with tanh(mean), plus its frozen image-encoder features."""

    def __init__(self, folder: str, device: str):
        from lerobot.policies.factory import make_pre_post_processors

        cfg = json.loads((Path(folder) / "config.json").read_text())
        if cfg["type"] == "sac_dagger_bc":
            from share.policies.sac_dagger import SACDaggerBCPolicy as cls
        elif cfg["type"] == "sac":
            from lerobot.policies.sac.modeling_sac import SACPolicy as cls
        else:
            raise ValueError(f"{folder}: unsupported policy type {cfg['type']!r}")
        self.policy = cls.from_pretrained(folder).to(device).eval()
        self.policy.config.device = device
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=folder)
        self.device = device
        self.image_keys = [k for k in self.policy.config.input_features if k.startswith("observation.images")]
        self.input_keys = list(self.policy.config.input_features)

    @torch.no_grad()
    def act(self, obs: dict) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
        batch = {k: obs[k].unsqueeze(0) if obs[k].ndim in (1, 3) else obs[k] for k in self.input_keys}
        batch = self.pre(batch)
        actor = self.policy.actor
        cache = actor.encoder.get_cached_image_features(batch) if self.image_keys else None
        enc = actor.encoder(batch, cache=cache)
        action = self.post(torch.tanh(actor.mean_layer(actor.network(enc))))
        feats = {k: cache[k][0].float().cpu().numpy() for k in self.image_keys} if cache else {}
        return action.squeeze(0).cpu(), feats


def _jpeg(frame: np.ndarray, quality: int) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def run_episode(net, policy: DeterministicPolicy, primitive: str, seed: int, realtime: bool, jpeg_quality: int):
    from lerobot.processor import TransitionKey
    from lerobot.utils.robot_utils import precise_sleep

    tr = net.reset(seed=seed)
    rec = {"state": [], "image": [], "fmap": [], "action": []}
    full, pose = [], []
    reason = None
    robot_name = next(iter(net.robot_dict))
    camera = next(iter(net.cameras.values()), None)
    while True:
        t0 = time.perf_counter()
        prim = net.active_primitive
        obs = tr[TransitionKey.OBSERVATION]
        if prim == primitive:
            action, feats = policy.act(obs)
            rec["state"].append(obs["observation.state"].detach().cpu().numpy().reshape(-1))
            img = obs[policy.image_keys[0]].detach().cpu().numpy().reshape(obs[policy.image_keys[0]].shape[-3:])
            rec["image"].append(np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8))
            rec["fmap"].append(feats[policy.image_keys[0]].astype(np.float16))
            rec["action"].append(action.numpy().astype(np.float32))
            pose.append([float(obs[f"{robot_name}.{ax}.ee_pos"]) for ax in ("x", "y", "z", "rx", "ry", "rz")])
            if camera is not None:
                full.append(_jpeg(np.asarray(camera.async_read()), jpeg_quality))
        else:
            action = torch.zeros(net.action_dim, dtype=torch.float32)
        tr = net.step(action)
        info = tr[TransitionKey.INFO]
        if info.get("transition_to", prim) != prim or tr[TransitionKey.DONE] or tr[TransitionKey.TRUNCATED]:
            if prim == primitive:
                reason = info.get("transition_reason") or ("time_limit" if tr[TransitionKey.TRUNCATED] else "done")
            if net.in_terminal:
                break
            tr = net.reset()
            if net.in_terminal:
                break
        if realtime:
            precise_sleep(1.0 / net.config.fps - (time.perf_counter() - t0))
    steps = len(rec["action"])
    arrays = {k: np.stack(v) if v else np.zeros((0,)) for k, v in rec.items()}
    meta = dict(success=reason == "success", steps=steps, duration_s=steps / float(net.config.fps),
                reason=str(reason))
    return arrays, meta, full, np.asarray(pose, dtype=np.float64)


def serve(args) -> None:
    queue = Path(args.queue)
    for sub in ("requests", "done"):
        (queue / sub).mkdir(parents=True, exist_ok=True)
    importlib.import_module(args.env_module)
    if args.mock:
        importlib.import_module(args.mock).use_mock_hardware(seed=args.mock_seed)
    from lerobot.envs import EnvConfig

    from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import ManipulationPrimitiveNet

    net, policies = None, {}
    try:
        while True:
            pending = [p for p in sorted((queue / "requests").glob("*.json"))
                       if not (queue / "done" / p.name).exists()]
            if not pending:
                if args.once:
                    return
                time.sleep(POLL_S)
                continue
            req = json.loads(pending[0].read_text())
            out = Path(req["out"])
            (out / "full").mkdir(parents=True, exist_ok=True)
            (out / "hidden").mkdir(parents=True, exist_ok=True)
            for ep in req["episodes"]:
                if (out / f"{ep['key']}.npz").exists():
                    continue
                if net is None:
                    net = ManipulationPrimitiveNet(EnvConfig.get_choice_class(args.env_type)())
                folder = req["policies"][ep["policy"]]
                if folder not in policies:
                    policies[folder] = DeterministicPolicy(folder, args.device)
                arrays, meta, full, pose = run_episode(net, policies[folder], req.get("primitive", "insert"),
                                                       int(ep["seed"]), not args.no_realtime, args.jpeg_quality)
                offsets = np.cumsum([0] + [len(b) for b in full]).astype(np.int64)
                _atomic_savez(out / "full" / f"{ep['key']}.npz",
                              jpeg=np.frombuffer(b"".join(full), dtype=np.uint8), offsets=offsets)
                _atomic_json(out / "hidden" / f"{ep['key']}.json", dict(tcp_pose_task_frame=pose.tolist(),
                                                                       seed=int(ep["seed"])))
                _atomic_savez(out / f"{ep['key']}.npz", **arrays, **{k: np.array(v) for k, v in meta.items()})
                print(f"[{req['id']}] {ep['key']}: {'success' if meta['success'] else meta['reason']} "
                      f"after {meta['duration_s']:.1f} s", flush=True)
            _atomic_json(queue / "done" / pending[0].name, dict(id=req["id"], finished=time.time()))
    finally:
        if net is not None:
            net.close()


def supervise(argv: list[str], queue: Path) -> None:
    cmd = [sys.executable, "-m", "share.scripts.rollout_runner", *[a for a in argv if a != "--supervise"]]
    queue.mkdir(parents=True, exist_ok=True)
    while True:
        rc = subprocess.call(cmd)
        if rc == 0:
            return
        _log(queue, f"runner exited with code {rc}; restarting in {RESTART_DELAY_S:.0f} s (unfinished episodes rerun)")
        time.sleep(RESTART_DELAY_S)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--queue", required=True)
    p.add_argument("--env_module", required=True, help="module that registers the env config on import")
    p.add_argument("--env_type", required=True, help="registered EnvConfig type, e.g. ur3e_han_insertion")
    p.add_argument("--device", default="cuda")
    p.add_argument("--jpeg_quality", type=int, default=90)
    p.add_argument("--mock", default=None, help="module with use_mock_hardware(seed) for headless dry runs")
    p.add_argument("--mock_seed", type=int, default=0)
    p.add_argument("--no_realtime", action="store_true", help="do not pace steps at the env fps (mock only)")
    p.add_argument("--once", action="store_true", help="exit when the queue is empty instead of waiting")
    p.add_argument("--supervise", action="store_true", help="restart the runner whenever it fails")
    args = p.parse_args()
    if args.supervise:
        supervise(sys.argv[1:], Path(args.queue))
    else:
        serve(args)


if __name__ == "__main__":
    main()
