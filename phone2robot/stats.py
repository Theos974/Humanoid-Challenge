"""Success-rate statistics. Episodes from the same test layout are not independent, so comparisons between two
policies resample layouts, not episodes."""
import json
import os

import numpy as np

from .config import WORK


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    mid = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (mid - half) / den, (mid + half) / den


def per_layout(labels):
    """Success per layout, pooled over several eval files (e.g. two training seeds)."""
    out = {}
    for lab in labels:
        with open(os.path.join(WORK, "eval", f"{lab}.json")) as f:
            for e in json.load(f)["episodes"]:
                out.setdefault(e["tag"], []).append(float(e["success"]))
    return out


def paired_diff(labels_a, labels_b, n_boot=20000, seed=0):
    """Mean difference in success (A - B) over shared layouts, with a 95% bootstrap interval over layouts."""
    a, b = per_layout(labels_a), per_layout(labels_b)
    tags = sorted(set(a) & set(b))
    d = np.array([np.mean(a[t]) - np.mean(b[t]) for t in tags])
    rng = np.random.RandomState(seed)
    boots = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(a=float(np.mean([np.mean(a[t]) for t in tags])), b=float(np.mean([np.mean(b[t]) for t in tags])),
                diff=float(d.mean()), ci=(float(lo), float(hi)), a_better=int((d > 0).sum()), b_better=int((d < 0).sum()),
                ties=int((d == 0).sum()), layouts=len(tags))
