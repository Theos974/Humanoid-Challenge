"""Turn one human clip (plate-relative hand path) into one robot demonstration in the simulator.

The clip decides where the bowl starts, which side of the rim is grasped, and the shape and timing of the reach and
carry. The controller knows the exact object positions and does the grasp and the placing."""
import numpy as np

from .config import CM_TO_M, PLATE_SIM, MIN_BOWL_PLATE_M, VIDEO_FPS, SIM_FPS
from .scene import write_scene, make_env, aabb_world, eef, Controller

XP, YP = PLATE_SIM


def cm_to_sim(uv):
    """Plate-relative cm (u right in the image, v away from the phone) -> table xy in metres."""
    u, v = uv
    return np.array([XP - v * CM_TO_M, YP - u * CM_TO_M])


def sim_to_cm(xy):
    return np.array([-(xy[1] - YP) / CM_TO_M, -(xy[0] - XP) / CM_TO_M])


def bowl_start_sim(bowl_cm):
    """Clip bowl position -> sim, pushed out of the plate if overlapping and clipped to the table."""
    b = cm_to_sim(bowl_cm)
    p = np.array(PLATE_SIM)
    d = b - p
    if np.linalg.norm(d) < MIN_BOWL_PLATE_M:
        b = p + d / max(np.linalg.norm(d), 1e-6) * MIN_BOWL_PLATE_M
    return np.clip(b, [-0.30, -0.30], [0.20, 0.30])


def smooth_path(P, sigma):
    """Gaussian smoothing along time (sigma in video frames), edges padded."""
    P = np.asarray(P, float)
    if len(P) < 5 or sigma <= 0:
        return P
    rad = int(3 * sigma)
    x = np.arange(-rad, rad + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    Pp = np.pad(P, ((rad, rad), (0, 0)), mode="edge")
    return np.stack([np.convolve(Pp[:, j], k, mode="valid") for j in range(P.shape[1])], 1)


def shape_path(H, n, p0, p1, max_dev=0.10):
    """Re-target a human path H (cm) to run from p0 to p1 (sim m) in n steps: keep its progress profile along the
    start-end chord and its sideways deviation from the chord (capped at max_dev metres)."""
    H = np.asarray(H, float)
    m = len(H)
    B = H[-1] - H[0]
    L2 = float(B @ B)
    if L2 < 4.0 or m < 3:
        tau, dev = np.linspace(0, 1, m), np.zeros_like(H)
    else:
        tau = np.maximum.accumulate(np.clip(((H - H[0]) @ B) / L2, 0, 1))
        dev = H - (H[0] + tau[:, None] * B)
    ti = np.linspace(0, m - 1, n)
    tau_n = np.interp(ti, np.arange(m), tau)
    tau_n[0], tau_n[-1] = 0.0, 1.0
    dev_n = np.stack([np.interp(ti, np.arange(m), dev[:, j]) for j in (0, 1)], 1)
    dsim = np.stack([-dev_n[:, 1], -dev_n[:, 0]], 1) * CM_TO_M
    mag = np.linalg.norm(dsim, axis=1, keepdims=True)
    dsim = dsim * np.minimum(1.0, max_dev / np.maximum(mag, 1e-9))
    return p0[None] + tau_n[:, None] * (p1 - p0)[None] + dsim


def _carry_end(C, goal_cm, thr_cm=8.0):
    """First frame (after the first 8) where the hand is within thr_cm of the goal; the rest is the human settling."""
    d = np.linalg.norm(np.asarray(C) - np.asarray(goal_cm), axis=1)
    return next((i for i in range(8, len(C)) if d[i] <= thr_cm), len(C) - 1)


def _steps(frames, slow, lo, hi):
    return int(np.clip(round(frames / VIDEO_FPS * SIM_FPS * slow), lo, hi))


def make_demo(rec, place="plate", target_xy=None, slow=1.0, reset_seed=None, nudge_std=0.0, nudge_seed=0):
    """Run one demonstration. rec: clip record (bowl_cm, grasp_cm, reach_cm, carry_cm, a, tg, release_cm).
    place='plate' puts the bowl on the plate; place='target' sets it down on the table at target_xy.
    Returns (info dict, demo arrays). info['success'] says whether the bowl ended where it should."""
    bowl_cm, g_cm = np.array(rec["bowl_cm"]), np.array(rec["grasp_cm"])
    env = make_env(write_scene(f"clip_{rec['tag']}", bowl_start_sim(bowl_cm), PLATE_SIM))
    obs, _ = env.reset(seed=int(rec["tag"]) if reset_seed is None else reset_seed)
    bmin, bmax = aabb_world(env, "akita_black_bowl_1")
    pmin, pmax = aabb_world(env, "plate_1")
    bc, pc = (bmin + bmax) / 2, (pmin + pmax) / 2
    rim = (bmax[0] - bmin[0]) / 2
    z_grip = bmax[2] - 0.02
    z_hover = bmax[2] + 0.10
    z_carry = pmax[2] + 0.04 + (z_grip - bmin[2])
    side = 1.0 if -(g_cm - bowl_cm)[0] >= 0 else -1.0          # which rim the hand went for
    off = np.array([0.0, side * (rim - 0.008)])
    p_grasp = bc[:2] + off
    if place == "plate":
        p_place, z_place, goal_cm = pc[:2] + off, pmax[2] + 0.012 + (z_grip - bmin[2]), (0.0, 0.0)
    else:
        p_place, z_place, goal_cm = np.asarray(target_xy) + off, z_grip + 0.008, rec["release_cm"]

    ctl = Controller(env, obs, nudge_std=nudge_std, nudge_seed=nudge_seed)
    z_home = eef(obs)[2]
    # reach: human path shape from the arm's start to above the grasp point
    n = _steps(rec["tg"] - rec["a"], slow, 25, 80)
    xy = shape_path(rec["reach_cm"], n, eef(obs)[:2], p_grasp)
    s = np.clip(np.arange(n) / (0.7 * (n - 1)), 0, 1)
    ctl.follow(np.c_[xy, z_home + (z_hover - z_home) * s * s * (3 - 2 * s)], -1)
    phases = dict(reach_end=ctl.t)
    ctl.goto([*p_grasp, z_hover], -1, tol=0.008, max_steps=60)
    ctl.goto([*p_grasp, z_grip], -1, tol=0.004, max_steps=80)
    ctl.hold(+1, 20)
    ctl.goto([*p_grasp, z_carry], +1, tol=0.01, max_steps=80)
    phases["carry_start"] = ctl.t
    # carry: human path shape from the grasp to above the place point
    C = np.asarray(rec["carry_cm"])
    ce = _carry_end(C, goal_cm)
    n = _steps(ce, slow, 30, 90)
    ctl.follow(np.c_[shape_path(C[:ce + 1], n, p_grasp, p_place), np.full(n, z_carry)], +1)
    phases["carry_end"] = ctl.t
    ctl.goto([*p_place, z_carry], +1, tol=0.008, max_steps=60)
    ctl.goto([*p_place, z_place], +1, tol=0.005, max_steps=80)
    ctl.hold(-1, 15)
    ctl.goto([*p_place, z_carry], -1, tol=0.012, max_steps=40)
    ctl.hold(-1, 5)
    phases["end"] = ctl.t

    bmin2, bmax2 = aabb_world(env, "akita_black_bowl_1")
    end = (bmin2 + bmax2) / 2
    if place == "plate":
        ok = bool(env._env.check_success())
    else:
        ok = bool(np.linalg.norm(end[:2] - target_xy) < 0.05 and bmin2[2] - bmin[2] < 0.02)
    # phases + clip frame numbers let the raw-frames experiment line up video frames with demo steps
    info = dict(tag=rec["tag"], success=ok, steps=ctl.t, bowl_xy=bc[:2].tolist(), side=side, phases=phases,
                ce_frames=int(ce), a=rec["a"], tg=rec["tg"], tr=rec.get("tr"))
    demo = ctl.demo()
    env.close()
    return info, demo


def demo_with_retry(rec, **kw):
    """Run at normal speed; if the bowl doesn't end in place, retry 1.5x slower (fast human motions can drop it)."""
    for slow in (1.0, 1.5):
        info, demo = make_demo(rec, slow=slow, **kw)
        if info["success"]:
            break
    info["slow"] = slow
    return info, demo


def scripted_record(tag, bowl_xy):
    """A 'robot-style' record at a given sim bowl position: straight reach and carry, fixed timing, rim on the
    plate's side. Used as the matched control for the human-derived demos."""
    bowl_cm = sim_to_cm(np.asarray(bowl_xy))
    side = 1.0 if bowl_xy[1] <= YP else -1.0
    g_cm = bowl_cm + np.array([-side * 5.0, 0.0])
    return dict(tag=tag, bowl_cm=bowl_cm.tolist(), grasp_cm=g_cm.tolist(), a=0, tg=60, release_cm=[0.0, 0.0],
                reach_cm=np.linspace(bowl_cm + np.array([0.0, 30.0]), g_cm, 40).tolist(),
                carry_cm=np.linspace(g_cm, np.array([0.0, 0.0]), 60).tolist())


def shifted(rec, rng, ok, r_range=(3.0, 12.0), tries=500):
    """Copy of rec with the bowl start (and grasp point) moved 3-12 cm in a random direction, staying where ok(pos)
    is true. Returns None if no such position is found."""
    for _ in range(tries):
        a, r = rng.uniform(0, 2 * np.pi), rng.uniform(*r_range)
        d = r * np.array([np.cos(a), np.sin(a)])
        if ok(np.asarray(rec["bowl_cm"]) + d):
            return dict(rec, bowl_cm=(np.asarray(rec["bowl_cm"]) + d).tolist(),
                        grasp_cm=(np.asarray(rec["grasp_cm"]) + d).tolist())
    return None
