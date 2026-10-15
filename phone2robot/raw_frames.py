"""Raw-frames experiment: the policy sees the real phone frames (not the simulator) and predicts where the robot's
gripper should be. Frames are time-aligned to the robot demo made from the same clip."""
import glob
import os

import cv2
import numpy as np

from .config import DATA


def find_video(tag):
    hits = sorted(glob.glob(os.path.join(DATA, "*", f"*_{tag}000_iOS*.MOV")))
    if not hits:
        raise FileNotFoundError(tag)
    return hits[0]


def prep_frame(bgr):
    """Crop to the table area, scale to 256 px high, centre on a black 256x256 canvas, RGB."""
    crop = bgr[150:min(1750, bgr.shape[0])]
    nw = max(1, int(round(crop.shape[1] * 256.0 / crop.shape[0])))
    img = cv2.resize(crop, (nw, 256), interpolation=cv2.INTER_AREA)
    out = np.zeros((256, 256, 3), np.uint8)
    x0 = (256 - nw) // 2
    out[:, x0:x0 + nw] = img[:, :256 - x0] if nw > 256 - x0 else img
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def load_frames(tag, upto):
    cap = cv2.VideoCapture(find_video(tag))
    frames = []
    while len(frames) <= upto:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(prep_frame(fr))
    return frames


def frame_for_step(i, meta, n_frames):
    """Video frame shown at demo step i: piecewise-linear time warp that maps the demo's phases (reach, grasp,
    carry, place) onto the clip's own events (hand enters, grasp, reaches plate, release)."""
    ph, a, tg, tr, ce = meta["phases"], meta["a"], meta["tg"], meta["tr"], meta["ce_frames"]
    re, cs, cend, end = ph["reach_end"], ph["carry_start"], ph["carry_end"], ph["end"]
    i = min(i, end)
    if i <= re:
        f = a + (tg - a) * (i / max(re, 1))
    elif i <= cs:
        f = tg
    elif i <= cend:
        f = tg + ce * ((i - cs) / max(cend - cs, 1))
    else:
        f = (tg + ce) + ((tr + 10) - (tg + ce)) * ((i - cend) / max(end - cend, 1))
    return int(round(min(max(f, 0), n_frames - 1)))


def last_needed_frame(meta):
    return int(meta["tr"]) + 12
