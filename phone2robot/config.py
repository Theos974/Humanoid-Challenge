"""Paths and fixed numbers shared by every stage."""
import os

DATA = os.path.expanduser(os.environ.get("P2R_DATA", "~/humanoid/data"))      # raw phone clips (.MOV), one folder per session
WORK = os.path.expanduser(os.environ.get("P2R_WORK", "~/humanoid/work"))      # tracks, demos, datasets, runs, eval results
HAND_MODEL = os.path.expanduser(os.environ.get("P2R_HAND_MODEL", "~/humanoid/models/hand_landmarker.task"))

VIDEO_FPS = 30.0          # phone clips
SIM_FPS = 20              # LIBERO control rate / dataset fps
PLATE_CM = 26.0           # real plate diameter, used as the scale reference in every clip

# Real (plate-relative, cm) -> simulator (table frame, m). u = right in the phone image, v = away from the phone.
CM_TO_M = 0.65 / 100      # the real table is scaled down to fit the Panda's reach
PLATE_SIM = (-0.09, 0.05)
MIN_BOWL_PLATE_M = 0.135  # bowl centre at least this far from the plate centre (no overlap)

TASK_ON = "pick up the black bowl and place it on the plate"
TASK_NEXT = "pick up the black bowl and place it next to the plate"

for d in ("scenes", "logs", "eval"):
    os.makedirs(os.path.join(WORK, d), exist_ok=True)
