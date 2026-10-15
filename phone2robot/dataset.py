"""Package demos (.npz from the teacher) as a LeRobotDataset in the same layout as the LIBERO datasets."""
import json
import os
import shutil

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .config import WORK, SIM_FPS, TASK_ON

FEATURES = {
    "observation.images.image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.images.image2": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (7,), "names": ["action"]},
}


def build(name, entries, threads=4):
    """entries: list of demo paths, or of {"file": path, "task": instruction} for per-episode instructions."""
    root = os.path.join(WORK, f"ds_{name}")
    if os.path.exists(root):
        shutil.rmtree(root)
    ds = LeRobotDataset.create(repo_id=f"local/{name}", fps=SIM_FPS, features=FEATURES, root=root, robot_type="panda",
                               use_videos=False, image_writer_threads=threads, image_writer_processes=0)
    files, frames = [], 0
    for e in entries:
        path, task = (e["file"], e["task"]) if isinstance(e, dict) else (e, TASK_ON)
        z = np.load(path, allow_pickle=True)
        for t in range(len(z["action"])):
            ds.add_frame({"observation.images.image": z["image"][t], "observation.images.image2": z["image2"][t],
                          "observation.state": z["state"][t], "action": z["action"][t], "task": task})
        ds.save_episode()
        files.append(path)
        frames += len(z["action"])
    ds.finalize()
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(dict(name=name, episodes=len(files), frames=frames, entries=entries), f)
    return root, len(files), frames
