"""Clip records -> robot demonstrations in the simulator (one .npz per demo) + a list file for build_dataset.py.

examples
  human paths, smoothed:     make_demos.py --clips clips_s1.json --name s10 --smooth 10
  robot-style control:       make_demos.py --clips clips_s1.json --name scripted --teacher scripted
  spread each clip x7:       make_demos.py --clips clips_s1.json --tags lists/spread10.json --name spread --smooth 10 --spread 7
  recovery (nudged) demos:   make_demos.py --clips clips_s1.json --tags lists/clean40.json --name nudge02 --smooth 10 --nudge 0.2
  'next to the plate' task:  make_demos.py --clips clips_s3_next.json --name next --smooth 10 --spread 2 --goal next"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from phone2robot.config import WORK, TASK_ON, TASK_NEXT
from phone2robot.teacher import (demo_with_retry, smooth_path, scripted_record, shifted, bowl_start_sim, cm_to_sim)

ap = argparse.ArgumentParser()
ap.add_argument("--clips", nargs="+", required=True, help="clip record JSONs (from process_clips.py)")
ap.add_argument("--tags", help="JSON list of clip tags (or demo paths named clip_<tag>.npz) to use; default all usable")
ap.add_argument("--name", required=True)
ap.add_argument("--teacher", choices=["human", "scripted"], default="human")
ap.add_argument("--smooth", type=float, default=0.0, help="Gaussian smoothing of the hand path, sigma in video frames")
ap.add_argument("--spread", type=int, default=1, help="demos per clip: the filmed bowl position + (spread-1) nearby ones")
ap.add_argument("--nudge", type=float, default=0.0, help="std of random pushes on the executed motion (recorded action stays clean)")
ap.add_argument("--reset-offset", type=int, default=0, help="different simulator reset -> a second, slightly different copy")
ap.add_argument("--goal", choices=["plate", "next"], default="plate")
ap.add_argument("--min-carry", type=int, default=30)
ap.add_argument("--min-reach", type=int, default=10)
args = ap.parse_args()

recs = {}
for p in args.clips:
    with open(p if os.path.exists(p) else os.path.join(WORK, p)) as f:
        for r in json.load(f):
            if r.get("valid"):
                recs[r["tag"]] = r
if args.tags:
    with open(args.tags) as f:
        tags = [os.path.basename(t)[5:-4] if t.endswith(".npz") else t for t in json.load(f)]
else:
    tags = [t for t, r in sorted(recs.items())
            if r["tr"] - r["tg"] >= args.min_carry and r["tg"] - r["a"] >= args.min_reach]

target = None
if args.goal == "next":     # the spot is where the hand let go, averaged over the clips (one fixed marker in the video)
    rel = np.array([recs[t]["release_cm"] for t in tags if recs[t]["release_cm"][1] > 15])
    target = cm_to_sim(np.median(rel, axis=0))
    tags = [t for t in tags if recs[t]["release_cm"][1] > 15]
instruction = TASK_NEXT if args.goal == "next" else TASK_ON


def in_area(b):
    """Shifted bowl starts stay where the robot can reach and away from the plate and the 'next to' spot."""
    ok = -6 <= b[0] <= 26 and -30 <= b[1] <= 32 and np.linalg.norm(b) >= 21.5
    return ok and (target is None or np.linalg.norm(cm_to_sim(b) - target) >= 0.10)


out_dir = os.path.join(WORK, f"demos_{args.name}")
os.makedirs(out_dir, exist_ok=True)
rng = np.random.RandomState(0)
entries, n_fail = [], 0
for k, tag in enumerate(tags):
    base = dict(recs[tag])
    if args.teacher == "scripted":
        base = dict(scripted_record(tag, bowl_start_sim(base["bowl_cm"])), release_cm=base.get("release_cm"))
    elif args.smooth > 0:
        base["reach_cm"] = smooth_path(base["reach_cm"], args.smooth).tolist()
        base["carry_cm"] = smooth_path(base["carry_cm"], args.smooth).tolist()
    for v in range(args.spread):
        path = os.path.join(out_dir, f"clip_{tag}_v{v}.npz" if args.spread > 1 else f"clip_{tag}.npz")
        if not os.path.exists(path):
            rec = base if v == 0 else shifted(base, rng, in_area)
            if rec is None:
                continue
            rec = dict(rec, tag=str(int(tag) + 100 * v))
            info, demo = demo_with_retry(rec, place="plate" if target is None else "target", target_xy=target,
                                         reset_seed=int(rec["tag"]) + args.reset_offset,
                                         nudge_std=args.nudge, nudge_seed=1000 * k + v)
            print(json.dumps(dict(clip=tag, variant=v, success=info["success"], steps=info["steps"])), flush=True)
            if not info["success"]:
                n_fail += 1
                continue
            np.savez_compressed(path, **demo, meta=json.dumps(info))
        entries.append(dict(file=path, task=instruction))
with open(os.path.join(WORK, f"lists_{args.name}.json"), "w") as f:
    json.dump(entries, f)
if target is not None:
    with open(os.path.join(WORK, f"target_{args.name}.json"), "w") as f:
        json.dump(dict(target_xy=target.tolist()), f)
print(f"{args.name}: {len(entries)} demos, {n_fail} failed")
