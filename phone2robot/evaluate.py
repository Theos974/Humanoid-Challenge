"""Closed-loop evaluation of a trained policy on held-out bowl positions.

Every episode records where the bowl ended up: on the plate (LIBERO's own success check), resting at the
'next to the plate' spot, or neither; and, with a distractor, whether the distractor was moved."""
import numpy as np
import gymnasium as gym
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvCfg
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import rollout

from .scene import write_scene, make_env, aabb_world, extra_name
from .stats import wilson

RENAME = {"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}


def load_policy(path, n_action_steps=10):
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.pretrained_path, cfg.device, cfg.n_action_steps = path, "cuda", n_action_steps
    env_cfg = LiberoEnvCfg(task="libero_spatial", task_ids=[2])
    policy = make_policy(cfg=cfg, env_cfg=env_cfg, rename_map=RENAME)
    policy.eval()
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=path, preprocessor_overrides={
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": RENAME}})
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
    return policy, (env_pre, env_post, pre, post)


class Tracker:
    """Called every step: is the bowl resting at the target spot (10+ consecutive steps)? how far did the
    distractor move?"""

    def __init__(self, n, target_xy=None, distractor=None):
        self.target = None if target_xy is None else np.asarray(target_xy)
        self.distractor = distractor
        self.z0, self.run, self.at_target = [None] * n, [0] * n, [False] * n
        self.d0, self.moved = [None] * n, [0.0] * n

    def __call__(self, venv):
        for i, env in enumerate(venv.envs):
            if self.target is not None:
                lo, hi = aabb_world(env, "akita_black_bowl_1")
                self.z0[i] = lo[2] if self.z0[i] is None else self.z0[i]
                near = np.linalg.norm((lo + hi)[:2] / 2 - self.target) < 0.05 and lo[2] - self.z0[i] < 0.02
                self.run[i] = self.run[i] + 1 if near else 0
                self.at_target[i] = self.at_target[i] or self.run[i] >= 10
            if self.distractor is not None:
                lo, hi = aabb_world(env, self.distractor)
                c = (lo + hi)[:2] / 2
                self.d0[i] = c if self.d0[i] is None else self.d0[i]
                self.moved[i] = max(self.moved[i], float(np.linalg.norm(c - self.d0[i])))


def evaluate(policy_path, layouts, prompt, seeds=3, batch=4, n_action_steps=10, max_steps=260,
             goal="plate", target_xy=None, distractor=None, render=None):
    """layouts: [{tag, bowl_xy, plate_xy}]. goal='plate' -> success = bowl on plate; goal='target' -> success =
    bowl resting at target_xy and not on the plate. distractor = (libero_object_type, (x, y)) or None.
    render(venv) is an optional per-step callback (for videos)."""
    policy, procs = load_policy(policy_path, n_action_steps)
    extras = [] if distractor is None else [distractor]
    dname = None if distractor is None else extra_name(distractor[0])
    episodes = []
    for b in range(0, len(layouts), batch):
        chunk = layouts[b:b + batch]

        def factory(L):
            def f():
                env = make_env(write_scene(f"eval_{L['tag']}", L["bowl_xy"], L["plate_xy"], extras), episode_length=max_steps)
                env.task_description = prompt
                env.task = "phone2robot"
                return env
            return f

        venv = gym.vector.SyncVectorEnv([factory(L) for L in chunk], autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)
        for s in range(seeds):
            tr = Tracker(len(chunk), target_xy, dname)
            cb = tr if render is None else (lambda v, tr=tr: (tr(v), render(v)))
            ro = rollout(venv, policy, *procs, seeds=[1000 * s + i for i in range(len(chunk))], render_callback=cb)
            on_plate = np.asarray(ro["success"]).any(axis=1)
            steps = (~np.asarray(ro["done"])).sum(axis=1) + 1
            for i, L in enumerate(chunk):
                ok = bool(on_plate[i]) if goal == "plate" else bool(tr.at_target[i] and not on_plate[i])
                episodes.append(dict(tag=L["tag"], seed=s, success=ok, steps=int(steps[i]), on_plate=bool(on_plate[i]),
                                     at_target=bool(tr.at_target[i]) if target_xy is not None else None,
                                     distractor_moved_m=round(tr.moved[i], 3) if dname else None))
        venv.close()
    k, n = sum(e["success"] for e in episodes), len(episodes)
    return dict(prompt=prompt, goal=goal, n=n, k=k, rate=k / n, ci=wilson(k, n), n_action_steps=n_action_steps,
                max_steps=max_steps, distractor=distractor, episodes=episodes)
