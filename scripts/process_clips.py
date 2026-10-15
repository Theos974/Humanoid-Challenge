"""Phone clips -> clip records (plate-relative hand paths, grasp/release, bowl start).
usage: python scripts/process_clips.py <folder with .MOV clips> <session name>
writes $P2R_WORK/tracks_<session>/<tag>.npz and $P2R_WORK/clips_<session>.json"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from phone2robot.config import WORK
from phone2robot.tracking import track_clip
from phone2robot.plate import last_frame, fit_plate
from phone2robot.extract import extract_clip

folder, session = sys.argv[1], sys.argv[2]
tdir = os.path.join(WORK, f"tracks_{session}")
os.makedirs(tdir, exist_ok=True)
records = []
for path in sorted(glob.glob(os.path.join(folder, "*.MOV"))):
    tag = os.path.basename(path)[9:15]                     # iPhone names: YYYYMMDD_HHMMSS000_iOS.MOV -> HHMMSS
    tpath = os.path.join(tdir, f"{tag}.npz")
    if os.path.exists(tpath):
        track = dict(np.load(tpath))
    else:
        track = track_clip(path)
        np.savez_compressed(tpath, **track)
    rec = extract_clip(tag, path, track, fit_plate(last_frame(path)))
    rec["file"] = os.path.basename(path)
    records.append(rec)
    print(f"{tag} valid={rec['valid']} {rec['why']}", flush=True)
with open(os.path.join(WORK, f"clips_{session}.json"), "w") as f:
    json.dump(records, f)
print(f"{session}: {sum(r['valid'] for r in records)} / {len(records)} clips usable")
