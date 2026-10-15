"""usage: python scripts/build_dataset.py <name> <list json from make_demos.py> [<more list jsons>...]
Several lists are concatenated (e.g. clean + nudged demos, or the 'on' and 'next to' tasks)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from phone2robot.config import WORK
from phone2robot.dataset import build

entries = []
for p in sys.argv[2:]:
    with open(p if os.path.exists(p) else os.path.join(WORK, p)) as f:
        entries += json.load(f)
root, n, frames = build(sys.argv[1], entries)
print(f"{root}: {n} episodes, {frames} frames")
