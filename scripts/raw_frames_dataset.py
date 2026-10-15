"""Dataset for the raw-frames experiment: each sample = a real phone frame -> the robot's gripper position a few steps
ahead in the demo made from that clip (absolute xyz) + gripper open/close. Both camera slots get the same frame and
the robot state is zeroed, so the policy can only use the video.
usage: python scripts/raw_frames_dataset.py <name> <list json of demos made with make_demos.py> [lead=5]"""
import json
import os
import shutil
import sys

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from phone2robot.config import WORK, SIM_FPS, TASK_ON
from phone2robot.dataset import FEATURES
from phone2robot.raw_frames import load_frames, frame_for_step, last_needed_frame

name, list_json = sys.argv[1], sys.argv[2]
lead = int(sys.argv[3]) if len(sys.argv) > 3 else 5          # the controller lags its target by ~5 steps
with open(list_json if os.path.isabs(list_json) else os.path.join(WORK, list_json)) as f:
    entries = json.load(f)
root = os.path.join(WORK, f"ds_{name}")
if os.path.exists(root):
    shutil.rmtree(root)
ds = LeRobotDataset.create(repo_id=f"local/{name}", fps=SIM_FPS, features=FEATURES, root=root, robot_type="panda",
                           use_videos=False, image_writer_threads=4)
n = 0
for e in entries:
    z = np.load(e["file"] if isinstance(e, dict) else e, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    tag = meta["tag"]
    frames = load_frames(tag, last_needed_frame(meta))
    T = len(z["action"])
    for i in range(T):
        a = np.zeros(7, np.float32)
        a[:3] = z["state"][min(i + 1 + lead, T - 1)][:3]
        a[6] = z["action"][i][6]
        f = frames[frame_for_step(i, meta, len(frames))]
        ds.add_frame({"observation.images.image": f, "observation.images.image2": f,
                      "observation.state": np.zeros(8, np.float32), "action": a, "task": TASK_ON})
    ds.save_episode()
    n += 1
ds.finalize()
print(f"{root}: {n} clips")
