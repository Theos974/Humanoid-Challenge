"""The simulated scene (LIBERO table with a plate and a bowl) and a position controller for the Panda arm."""
import os
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv

from .config import WORK

BDDL = """(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  (:language Pick the akita black bowl and place it on the plate)
    (:regions
      (plate_region
          (:target main_table)
          (:ranges (
              ({px0:.4f} {py0:.4f} {px1:.4f} {py1:.4f})
            )
          )
      )
      (bowl_region
          (:target main_table)
          (:ranges (
              ({bx0:.4f} {by0:.4f} {bx1:.4f} {by1:.4f})
            )
          )
      )
    )
  (:fixtures
    main_table - table
  )
  (:objects
    akita_black_bowl_1 - akita_black_bowl
    plate_1 - plate
  )
  (:obj_of_interest
    akita_black_bowl_1
    plate_1
  )
  (:init
    (On akita_black_bowl_1 main_table_bowl_region)
    (On plate_1 main_table_plate_region)
  )
  (:goal
    (And (On akita_black_bowl_1 plate_1))
  )
)
"""


def write_scene(name, bowl_xy, plate_xy, extras=(), eps=0.003):
    """Write a BDDL file with the bowl and plate at fixed table positions, plus optional distractors
    extras = [(libero_object_type, (x, y)), ...]. Returns the file path."""
    (bx, by), (px, py) = bowl_xy, plate_xy
    txt = BDDL.format(px0=px - eps, py0=py - eps, px1=px + eps, py1=py + eps,
                      bx0=bx - eps, by0=by - eps, bx1=bx + eps, by1=by + eps)
    regions, objects, inits = "", "", ""
    for i, (otype, (x, y)) in enumerate(extras):
        name_i = f"akita_black_bowl_{i + 2}" if otype == "akita_black_bowl" else f"{otype}_{i + 1}"
        reg = f"extra{i}_region"
        regions += (f"      ({reg}\n          (:target main_table)\n          (:ranges (\n"
                    f"              ({x - eps:.4f} {y - eps:.4f} {x + eps:.4f} {y + eps:.4f})\n            )\n          )\n      )\n")
        if otype == "akita_black_bowl":       # LIBERO needs objects of the same type declared on one line
            txt = txt.replace("    akita_black_bowl_1 - akita_black_bowl\n", f"    akita_black_bowl_1 {name_i} - akita_black_bowl\n")
        else:
            objects += f"    {name_i} - {otype}\n"
        inits += f"    (On {name_i} main_table_{reg})\n"
    txt = txt.replace("    )\n  (:fixtures", regions + "    )\n  (:fixtures")
    txt = txt.replace("    plate_1 - plate\n", "    plate_1 - plate\n" + objects)
    txt = txt.replace("    (On plate_1 main_table_plate_region)\n", "    (On plate_1 main_table_plate_region)\n" + inits)
    path = os.path.join(WORK, "scenes", f"{name}.bddl")
    with open(path, "w") as f:
        f.write(txt)
    return path


def extra_name(otype, i=0):
    return f"akita_black_bowl_{i + 2}" if otype == "akita_black_bowl" else f"{otype}_{i + 1}"


def make_env(bddl, episode_length=700):
    """A LeRobot LiberoEnv that loads our BDDL file instead of a benchmark task."""
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    env = LiberoEnv(task_suite=suite, task_id=2, task_suite_name="libero_spatial", init_states=False,
                    obs_type="pixels_agent_pos", episode_length=episode_length)
    env._task_bddl_file = bddl
    return env


def aabb_world(env, body_prefix):
    """World-frame bounding box (min, max) of all collision geoms of the body whose name starts with body_prefix."""
    sim = env._env.env.sim
    m, d = sim.model, sim.data
    root = [i for i in range(m.nbody) if m.body(i).name.startswith(body_prefix)][0]
    ids, changed = {root}, True
    while changed:
        changed = False
        for b in range(m.nbody):
            if m.body_parentid[b] in ids and b not in ids:
                ids.add(b)
                changed = True
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in ids and m.geom_contype[g] + m.geom_conaffinity[g] > 0:
            c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
            R = d.geom_xmat[g].reshape(3, 3)
            pts += [d.geom_xpos[g] + R @ (c + h * s) for s in corners]
    pts = np.array(pts)
    return pts.min(0), pts.max(0)


def eef(obs):
    return np.array(obs["robot_state"]["eef"]["pos"], dtype=float).reshape(-1)[:3]


def quat2axisangle(q):
    """(x, y, z, w) quaternion -> axis-angle, matching LeRobot's LIBERO processor."""
    q = np.asarray(q, dtype=np.float32)
    w = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(max(1.0 - w * w, 0.0))
    if den < 1e-10:
        return np.zeros(3, np.float32)
    return (q[:3] / den * 2.0 * np.arccos(w)).astype(np.float32)


def policy_obs(obs):
    """Raw env observation -> what is stored in the dataset: both images rotated 180 deg (LeRobot convention) + 8-d state."""
    px, rs = obs["pixels"], obs["robot_state"]
    pos = np.asarray(rs["eef"]["pos"], dtype=np.float32).reshape(-1)[:3]
    quat = np.asarray(rs["eef"]["quat"], dtype=np.float32).reshape(-1)[:4]
    grip = np.asarray(rs["gripper"]["qpos"], dtype=np.float32).reshape(-1)[:2]
    return dict(image=np.ascontiguousarray(px["image"][::-1, ::-1]),
                image2=np.ascontiguousarray(px["image2"][::-1, ::-1]),
                state=np.concatenate([pos, quat2axisangle(quat), grip]).astype(np.float32))


class Controller:
    """Drives the end effector to position targets with a P-controller and records (observation, action) pairs.

    nudge_std > 0 adds Gaussian noise to the executed translation while the recorded action stays the clean,
    correcting one (DART-style recovery data)."""

    def __init__(self, env, obs, nudge_std=0.0, nudge_seed=0, keep_frames=False):
        self.env, self.obs = env, obs
        self.nudge_std, self.rng = nudge_std, np.random.RandomState(nudge_seed)
        self.keep_frames, self.frames, self.log, self.t = keep_frames, [], [], 0

    def step(self, a):
        a = np.clip(np.asarray(a, dtype=np.float32), -1, 1)
        p = policy_obs(self.obs)
        self.log.append((p, a.copy()))
        if self.keep_frames:
            self.frames.append(p["image"])
        a_exec = a.copy()
        if self.nudge_std > 0:
            a_exec[:3] = np.clip(a_exec[:3] + self.rng.normal(0, self.nudge_std, 3), -1, 1)
        self.obs, _, term, _, _ = self.env.step(a_exec)
        self.t += 1
        return term

    def _action(self, target, grip, gain=2.5):
        a = np.zeros(7, np.float32)
        a[:3] = np.clip(gain * (np.asarray(target, float) - eef(self.obs)) / 0.05, -1, 1)
        a[6] = grip
        return a

    def follow(self, path, grip):
        """One target per control step."""
        for target in path:
            self.step(self._action(target, grip))

    def goto(self, target, grip, max_steps=120, tol=0.006):
        for _ in range(max_steps):
            done = np.linalg.norm(np.asarray(target, float) - eef(self.obs)) < tol
            self.step(self._action(target, grip))
            if done:
                break

    def hold(self, grip, steps):
        for _ in range(steps):
            a = np.zeros(7, np.float32)
            a[6] = grip
            self.step(a)

    def demo(self):
        return dict(image=np.stack([p["image"] for p, _ in self.log]), image2=np.stack([p["image2"] for p, _ in self.log]),
                    state=np.stack([p["state"] for p, _ in self.log]), action=np.stack([a for _, a in self.log]))
