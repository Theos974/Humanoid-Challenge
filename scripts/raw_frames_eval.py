"""Raw-frames experiment, evaluation: play a held-out clip's real frames into the policy and move the simulated arm to
the predicted gripper positions. Success = bowl on the plate.
'oracle' instead of a policy replays the demo's own targets: the best this kind of label can do.
usage: python scripts/raw_frames_eval.py <label> <run name | checkpoint dir | oracle> <list json of held-out demos>
         [--seeds 3] [--same-reset] [--lead 5] [--video-dir videos/raw_frames]"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from phone2robot.config import WORK, TASK_ON, PLATE_SIM
from phone2robot.scene import write_scene, make_env, eef
from phone2robot.raw_frames import load_frames, frame_for_step, last_needed_frame
from phone2robot.stats import wilson

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("policy")
ap.add_argument("demos")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--same-reset", action="store_true", help="seed 0 uses the demo's own simulator reset")
ap.add_argument("--lead", type=int, default=5)
ap.add_argument("--video-dir")
a = ap.parse_args()

oracle = a.policy == "oracle"
if not oracle:
    from phone2robot.evaluate import load_policy
    path = a.policy if os.path.isdir(a.policy) else os.path.join(WORK, "runs", a.policy, "checkpoints/last/pretrained_model")
    policy, (_, _, pre, post) = load_policy(path)
with open(a.demos if os.path.isabs(a.demos) else os.path.join(WORK, a.demos)) as f:
    demos = [e["file"] if isinstance(e, dict) else e for e in json.load(f)]

results = []
for d in demos:
    z = np.load(d, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    tag, state, act = meta["tag"], z["state"], z["action"]
    frames = load_frames(tag, last_needed_frame(meta))
    end = meta["phases"]["end"]
    for s in range(a.seeds):
        env = make_env(write_scene(f"raw_{tag}", meta["bowl_xy"], PLATE_SIM), episode_length=4000)
        obs, _ = env.reset(seed=int(tag) if (a.same_reset and s == 0) else 1000 * s + int(tag) % 997)
        if not oracle:
            policy.reset()
            torch.manual_seed(1000 * s + 7)
        success, done_at, errs, vid = False, None, [], []
        for p in range(min(end + 80, 900)):
            f = frame_for_step(p, meta, len(frames))
            ref = state[min(p + 1 + a.lead, len(state) - 1)][:3]
            if oracle:
                tgt, g = ref, act[min(p, len(act) - 1)][6]
            else:
                img = torch.from_numpy(frames[f]).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
                batch = {"observation.images.image": img, "observation.images.image2": img.clone(),
                         "observation.state": torch.zeros(1, 8), "task": [TASK_ON]}
                with torch.inference_mode():
                    out = post(policy.select_action(pre(batch))).detach().cpu().numpy().reshape(-1)
                tgt, g = out[:3], out[6]
            errs.append(float(np.linalg.norm(np.asarray(tgt) - ref)))
            u = np.zeros(7, np.float32)
            u[:3] = np.clip(2.5 * (np.asarray(tgt) - eef(obs)) / 0.05, -1, 1)
            u[6] = 1.0 if g > 0 else -1.0
            obs, _, _, _, info = env.step(u)
            if a.video_dir and s == 0:
                vid.append((frames[f], np.array(env.render())))
            if info.get("is_success", False) and not success:
                success, done_at = True, p
            if success and p > done_at + 12:
                break
        env.close()
        results.append(dict(tag=tag, seed=s, success=success, mean_target_err_cm=round(100 * float(np.mean(errs)), 2)))
        print(f"{tag} seed {s}: {'success' if success else 'fail'}  mean target error {100 * np.mean(errs):.1f} cm", flush=True)
        if vid:
            os.makedirs(a.video_dir, exist_ok=True)
            vw = cv2.VideoWriter(os.path.join(a.video_dir, f"{a.label}_{tag}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (1024, 512))
            for real, sim in vid:
                vw.write(cv2.cvtColor(np.hstack([cv2.resize(real, (512, 512)), cv2.resize(sim, (512, 512))]), cv2.COLOR_RGB2BGR))
            vw.release()

k, n = sum(r["success"] for r in results), len(results)
res = dict(label=a.label, policy=a.policy, n=n, k=k, rate=k / n, ci=wilson(k, n), lead=a.lead, episodes=results,
           mean_target_err_cm=float(np.mean([r["mean_target_err_cm"] for r in results])))
with open(os.path.join(WORK, "eval", f"{a.label}.json"), "w") as f:
    json.dump(res, f, indent=1)
print(f"{a.label}: {k}/{n} = {100 * k / n:.1f}%  mean target error {res['mean_target_err_cm']:.1f} cm")
