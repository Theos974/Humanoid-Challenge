"""Render rollouts of a trained policy to mp4.
  one instruction:    videos.py <run> --layouts splits/test_layouts.json --tags 120721,231650 --out videos/
  two side by side:   videos.py <run> --layouts splits/test_layouts_two.json --tags a212552 --prompts on,next --out videos/"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from phone2robot.config import WORK, TASK_ON, TASK_NEXT
from phone2robot.evaluate import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("policy")
ap.add_argument("--layouts", required=True)
ap.add_argument("--tags", required=True)
ap.add_argument("--prompts", default="on", help="on, next, or on,next for a side-by-side video")
ap.add_argument("--target", help="JSON with target_xy (to score the 'next to' prompt)")
ap.add_argument("--out", default="videos")
a = ap.parse_args()

policy = a.policy if os.path.isdir(a.policy) else os.path.join(WORK, "runs", a.policy, "checkpoints/last/pretrained_model")
with open(a.layouts) as f:
    by_tag = {l["tag"]: l for l in json.load(f)}
layouts = [by_tag[t] for t in a.tags.split(",")]
prompts = {"on": TASK_ON, "next": TASK_NEXT}
target = None
if a.target:
    with open(a.target) as f:
        target = json.load(f)["target_xy"]


def record(p):
    frames = []
    res = evaluate(policy, layouts, prompts[p], seeds=1, batch=len(layouts), target_xy=target,
                   goal="target" if p == "next" else "plate",
                   render=lambda v: frames.append(np.stack([np.array(e.render()) for e in v.envs])))
    F = np.stack(frames)                                    # (T, n_layouts, H, W, 3)
    for i, ep in enumerate(res["episodes"]):                # hold the last frame once an episode ends (no auto-reset)
        F[ep["steps"]:, i] = F[ep["steps"] - 1, i]
    return F, res["episodes"]


def caption(img, text):
    bar = np.zeros((40, img.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return np.vstack([bar, img])


runs = [(p, *record(p)) for p in a.prompts.split(",")]
os.makedirs(a.out, exist_ok=True)
for i, L in enumerate(layouts):
    name = os.path.join(a.out, f"{os.path.basename(a.policy)}_{L['tag']}_{'_'.join(p for p, _, _ in runs)}.mp4")
    T = max(len(F) for _, F, _ in runs)
    vw = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*"mp4v"), 20, (512 * len(runs), 552))
    for t in range(T):
        tiles = [caption(cv2.resize(F[min(t, len(F) - 1), i], (512, 512)), f'"...{prompts[p][27:]}"') for p, F, _ in runs]
        vw.write(cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR))
    vw.release()
    print(name, " ".join(f"{p}:{'ok' if eps[i]['success'] else 'fail'}" for p, _, eps in runs))
