"""Curation experiment: keep only clips whose hand track looks clean, up to 15 per area around the plate, plus a
random selection with the same count per area as the control. Uses track quality only, never test results.
usage: python scripts/select_clips.py <clip jsons...> --tracks <tracks dirs...> --out-prefix cur"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from phone2robot.config import WORK
from phone2robot.teacher import bowl_start_sim

ap = argparse.ArgumentParser()
ap.add_argument("clips", nargs="+")
ap.add_argument("--tracks", nargs="+", required=True)
ap.add_argument("--exclude", help="JSON list of held-out tags")
ap.add_argument("--out-prefix", default="cur")
ap.add_argument("--per-area", type=int, default=15)
a = ap.parse_args()

recs = {}
for p in a.clips:
    with open(p) as f:
        recs.update({r["tag"]: r for r in json.load(f) if r.get("valid")})
held_out = set(json.load(open(a.exclude))) if a.exclude else set()
tracks = {os.path.basename(f)[-10:-4]: f for d in a.tracks for f in glob.glob(os.path.join(d, "*.npz"))}


def area(rec):
    x = bowl_start_sim(rec["bowl_cm"])[0]
    return "behind" if x < -0.20 else ("beside" if x < 0.05 else "front")


def quality(rec, pinch):
    """Pinch contrast, how steadily the pinch stays closed while carrying, path straightness, grasp point near the
    rim, and whether the carry actually ends at the plate."""
    a, tg, tr = rec["a"], rec["tg"], rec["tr"]
    ok = ~np.isnan(pinch)
    p = np.interp(np.arange(len(pinch)), np.where(ok)[0], pinch[ok])
    open_ref, closed_ref = np.median(p[a:max(a + 1, tg - 3)]), np.median(p[tg + 3:max(tg + 4, tr - 3)])
    thr = 0.5 * (open_ref + closed_ref)
    C = np.asarray(rec["carry_cm"])
    d = np.linalg.norm(C, axis=1)
    ce = next((i for i in range(8, len(C)) if d[i] <= 8.0), len(C) - 1)
    chord = np.linalg.norm(C[ce] - C[0])
    length = np.linalg.norm(np.diff(C[:ce + 1], axis=0), axis=1).sum()
    return dict(margin=(open_ref - closed_ref) / max(open_ref, 1e-6), closed_frac=float((p[tg:tr + 1] < thr).mean()),
                straightness=length / max(chord, 1e-6),
                grasp_to_bowl_cm=float(np.linalg.norm(np.subtract(rec["grasp_cm"], rec["bowl_cm"]))),
                closest_to_plate_cm=float(d.min()), carry_frames=tr - tg, reach_frames=tg - a)


def passes(q):
    return (q["margin"] >= 0.25 and q["closed_frac"] >= 0.85 and 3 <= q["grasp_to_bowl_cm"] <= 13
            and q["closest_to_plate_cm"] <= 10 and q["carry_frames"] >= 30 and q["reach_frames"] >= 10
            and q["straightness"] <= 2.0)


rows = []
for tag, rec in sorted(recs.items()):
    if tag in held_out or tag not in tracks:
        continue
    q = quality(rec, np.load(tracks[tag])["pinch"].astype(float))
    rows.append(dict(tag=tag, area=area(rec), passes=passes(q), **q))
rng = np.random.RandomState(0)
curated, random_ctl = [], []
for ar in ("behind", "beside", "front"):
    pool = [r for r in rows if r["area"] == ar]
    good = sorted([r for r in pool if r["passes"]], key=lambda r: (r["straightness"], -r["margin"]))[:a.per_area]
    curated += [r["tag"] for r in good]
    random_ctl += [pool[i]["tag"] for i in rng.choice(len(pool), size=len(good), replace=False)]
    print(f"{ar:7s} clips {len(pool):3d}  pass {sum(r['passes'] for r in pool):3d}  selected {len(good)}")
for name, tags in (("", curated), ("_random", random_ctl)):
    with open(os.path.join(WORK, f"tags_{a.out_prefix}{name}.json"), "w") as f:
        json.dump(tags, f)
with open(os.path.join(WORK, f"quality_{a.out_prefix}.json"), "w") as f:
    json.dump(rows, f, indent=1)
