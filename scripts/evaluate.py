"""usage: python scripts/evaluate.py <label> <run name or checkpoint dir> --layouts splits/test_layouts.json [options]
writes $P2R_WORK/eval/<label>.json and prints the success rate with a 95% Wilson interval."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from phone2robot.config import WORK, TASK_ON
from phone2robot.evaluate import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("policy", help="run name under $P2R_WORK/runs, or a checkpoint directory")
ap.add_argument("--layouts", required=True)
ap.add_argument("--prompt", default=TASK_ON)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--n-action-steps", type=int, default=10, help="actions executed before the policy re-plans")
ap.add_argument("--goal", choices=["plate", "next"], default="plate")
ap.add_argument("--target", help="JSON with target_xy, needed for --goal next (also recorded for --goal plate)")
ap.add_argument("--distractor", help="libero_object_type:x:y, e.g. glazed_rim_porcelain_ramekin:-0.05:-0.27")
a = ap.parse_args()

policy = a.policy if os.path.isdir(a.policy) else os.path.join(WORK, "runs", a.policy, "checkpoints/last/pretrained_model")
with open(a.layouts) as f:
    layouts = json.load(f)
target = None
if a.target:
    with open(a.target) as f:
        target = json.load(f)["target_xy"]
dist = None
if a.distractor:
    t, x, y = a.distractor.split(":")
    dist = (t, (float(x), float(y)))
res = evaluate(policy, layouts, a.prompt, seeds=a.seeds, n_action_steps=a.n_action_steps,
               goal="plate" if a.goal == "plate" else "target", target_xy=target, distractor=dist)
res.update(label=a.label, policy=policy)
with open(os.path.join(WORK, "eval", f"{a.label}.json"), "w") as f:
    json.dump(res, f, indent=1)
lo, hi = res["ci"]
print(f"{a.label}: {res['k']}/{res['n']} = {100 * res['rate']:.1f}%  (95% CI {100 * lo:.1f}-{100 * hi:.1f}%)")
if target is not None:
    eps = res["episodes"]
    print(f"  bowl on plate {sum(e['on_plate'] for e in eps)}, at 'next to' spot {sum(e['at_target'] for e in eps)}, of {len(eps)}")
