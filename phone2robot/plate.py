"""Find the plate in a clip's last frame. The plate (26 cm) is the ruler: it gives the pixel-to-cm scale and the
origin of every clip's coordinates, so the phone did not need to stay in the same place between sessions."""
import cv2
import numpy as np


def last_frame(path):
    cap = cv2.VideoCapture(path)
    last = None
    while True:
        ok, fr = cap.read()
        if not ok:
            return last
        last = fr


def edge_points(img):
    """Pixels on the boundary of bright regions (plate, bowl) that touch the dark mat."""
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    bright = (g > 110).astype(np.uint8)
    dark = (g < 80).astype(np.uint8)
    edge = bright - cv2.erode(bright, np.ones((3, 3), np.uint8))
    near_dark = cv2.dilate(dark, np.ones((9, 9), np.uint8))
    ys, xs = np.nonzero((edge > 0) & (near_dark > 0))
    m = (ys > 250) & (ys < 1500) & (xs < 950)                     # the plate is never at the frame borders
    return np.stack([xs[m], ys[m]], 1).astype(np.float32)


def vote_ellipse(P, a_rng, r_rng, cx_rng, cy_rng, tol=0.03):
    """Brute-force search for the axis-aligned ellipse (horizontal semi-axis a, vertical r*a) passing through the
    most edge points. The bowl's edge points do not fit any single plate-sized ellipse, so they are outvoted."""
    if len(P) > 2500:
        P = P[np.random.RandomState(0).choice(len(P), 2500, replace=False)]
    best = (-1, None)
    for a in a_rng:
        for r in r_rng:
            for cx in cx_rng:
                dx = (P[:, 0][None, :] - cx) / a
                cy = np.asarray(cy_rng)[:, None]
                dy = (P[:, 1][None, :] - cy) / (a * r)
                score = (np.abs(np.sqrt(dx ** 2 + dy ** 2) - 1) < tol).sum(1)
                k = int(score.argmax())
                if score[k] > best[0]:
                    best = (int(score[k]), (cx, float(cy[k, 0]), a, r))
    return best


def coverage(P, e, tol):
    """Fraction of the ellipse's 10-degree sectors that contain at least one inlier."""
    cx, cy, a, r = e
    rho = np.sqrt(((P[:, 0] - cx) / a) ** 2 + ((P[:, 1] - cy) / (a * r)) ** 2)
    inl = np.abs(rho - 1) < tol
    ang = np.degrees(np.arctan2((P[inl, 1] - cy) / r, P[inl, 0] - cx)) % 360
    return len(np.unique((ang // 10).astype(int))) / 36.0, int(inl.sum())


def fit_plate(img):
    """Coarse grid vote, then a fine one around the best. Returns dict(cx, cy, a, r, cov, ok) in pixels."""
    P = edge_points(img)
    _, (cx, cy, a, r) = vote_ellipse(P, np.arange(300, 441, 10), [0.62, 0.68, 0.74, 0.80, 0.86],
                                     np.arange(150, 561, 10), np.arange(400, 1161, 10))
    _, e = vote_ellipse(P, np.arange(a - 10, a + 11, 2.5), np.arange(r - 0.04, r + 0.041, 0.01),
                        np.arange(cx - 10, cx + 11, 2.5), np.arange(cy - 10, cy + 11, 2.5), tol=0.02)
    cov, n_inliers = coverage(P, e, tol=0.02)
    cx, cy, a, r = e
    return dict(cx=float(cx), cy=float(cy), a=float(a), r=float(r), cov=cov, n_inliers=n_inliers,
                ok=bool(cov >= 0.30 and 300 <= a <= 420))
