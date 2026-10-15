"""From a hand track + plate fit to one clip record in plate-relative centimetres:
grasp and release frames (from the pinch signal), where the bowl started, and the reach and carry paths of the
point between thumb and index fingertips."""
import cv2
import numpy as np

from .config import PLATE_CM


def first_last_frames(path):
    cap = cv2.VideoCapture(path)
    first = last = None
    while True:
        ok, fr = cap.read()
        if not ok:
            return first, last
        first = fr if first is None else first
        last = fr


def fill_gaps(x, max_gap=12):
    """Linear interpolation over NaN gaps of at most max_gap frames; longer gaps and the ends stay NaN."""
    x = x.copy()
    isn = np.isnan(x) if x.ndim == 1 else np.isnan(x).any(1)
    good = ~isn
    if good.sum() < 2:
        return x
    idx = np.arange(len(x))
    cols = [x] if x.ndim == 1 else [x[:, k] for k in range(x.shape[1])]
    out = []
    for col in cols:
        f = np.interp(idx, idx[good], col[good])
        run = 0
        for i in range(len(x)):
            if isn[i]:
                run += 1
            else:
                if run > max_gap:
                    f[i - run:i] = np.nan
                run = 0
        f[:np.argmax(good)] = np.nan
        f[len(good) - np.argmax(good[::-1]):] = np.nan
        out.append(f)
    return out[0] if x.ndim == 1 else np.stack(out, 1)


def median_filter(x, k=5):
    pad = np.pad(x, (k // 2, k // 2), mode="edge")
    return np.array([np.nanmedian(pad[i:i + k]) for i in range(len(x))])


def find_grasp(s, a, b):
    """Grasp/release = the longest stretch (gaps of up to 3 frames allowed) where the pinch is below the midpoint
    between the open level (first frames of the hand in view) and the closed level (15th percentile)."""
    seg = s[a:b + 1]
    open_lvl = np.nanmedian(s[a:a + 10])
    closed_lvl = np.nanpercentile(seg[~np.isnan(seg)], 15)
    thr = 0.5 * (open_lvl + closed_lvl)
    below = (s < thr) & ~np.isnan(s)
    runs, i = [], a
    while i <= b:
        if not below[i]:
            i += 1
            continue
        j, gap, k = i, 0, i
        while k <= b and gap <= 3:
            j, gap = (k, 0) if below[k] else (j, gap + 1)
            k += 1
        runs.append((i, j))
        i = j + 1
    if not runs:
        return None
    tg, tr = max(runs, key=lambda r: r[1] - r[0])
    return tg, tr


def bowl_start_px(first, last, plate):
    """The bowl is the biggest region that changed between the first and last frame, outside the plate."""
    blur = lambda im: cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (9, 9), 0)
    m = (cv2.absdiff(blur(first), blur(last)) > 35).astype(np.uint8)
    ell = np.zeros_like(m)
    cv2.ellipse(ell, (int(plate["cx"]), int(plate["cy"])), (int(plate["a"]), int(plate["a"] * plate["r"])), 0, 0, 360, 1, -1)
    m[ell > 0] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    return cen[1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))]


def to_cm(xy, plate):
    """Pixels -> plate-relative cm (u right, v away from the phone). The plate ellipse gives a separate scale for
    each image axis, which corrects for the camera looking at the table at an angle."""
    sx = plate["a"] / (PLATE_CM / 2)
    sy = plate["a"] * plate["r"] / (PLATE_CM / 2)
    xy = np.asarray(xy, float)
    return np.stack([(xy[..., 0] - plate["cx"]) / sx, -(xy[..., 1] - plate["cy"]) / sy], -1)


def extract_clip(tag, video_path, track, plate):
    """Returns a record; record['valid'] is False (with a reason in 'why') if the clip can't be used."""
    rec = dict(tag=tag, plate=plate, valid=False, why="")
    det = track["det"]
    idx = np.where(det)[0]
    if len(idx) < 20:
        return dict(rec, why="too few detections")
    a, b = int(idx[0]), int(idx[-1])
    grip_px = fill_gaps((track["pts"][:, 4] + track["pts"][:, 8]) / 2)
    ev = find_grasp(median_filter(fill_gaps(track["pinch"])), a, b)
    if ev is None:
        return dict(rec, why="no grasp found")
    tg, tr = ev
    rec.update(a=a, b=b, tg=int(tg), tr=int(tr))
    bpx = bowl_start_px(*first_last_frames(video_path), plate)
    if bpx is None:
        return dict(rec, why="bowl start not found")
    g = to_cm(grip_px, plate)
    if np.isnan(g[tg]).any() or np.isnan(g[tr]).any():
        return dict(rec, why="hand not tracked at grasp or release")
    rec.update(bowl_cm=to_cm(bpx, plate).tolist(), grasp_cm=g[tg].tolist(), release_cm=g[tr].tolist(),
               reach_cm=g[a:tg + 1].tolist(), carry_cm=g[tg:tr + 1].tolist())
    lost = int(np.isnan(g[a:tr + 1]).any(1).sum())
    rec["valid"] = lost == 0
    rec["why"] = "" if lost == 0 else f"{lost} untracked frames between entry and release"
    return rec
