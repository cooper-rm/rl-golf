"""Training helpers for the swing task (stable-baselines3 PPO).

Thin wrappers so the notebook and scripts share the same setup: build a vectorized
env, a sensibly-configured PPO, and a callback that records a swing clip every N
steps so you can literally watch the policy improve.
"""

from __future__ import annotations

import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from rl_golf.body import BodyParams
from rl_golf.swing_env import SwingEnv
from rl_golf.viz import rollout, save_mp4


def _make_env(body):
    def _factory():
        return SwingEnv(body=body)
    return _factory


def make_vec_env(body: BodyParams | None = None, n_envs: int = 8,
                 normalize: bool = False, subprocess: bool = True):
    """Vectorized training env.

    Observations are already scaled to O(1) inside SwingEnv, so VecNormalize is
    off by default — that keeps the policy consistent between training and the
    raw-env rollout/eval helpers (no obs-stats mismatch). Set normalize=True only
    if you also reuse the saved VecNormalize stats at rollout time.

    subprocess=True runs envs in parallel processes (much faster for MuJoCo on a
    multicore CPU). Use subprocess=False inside notebooks if you hit spawn issues.
    """
    factories = [_make_env(body) for _ in range(n_envs)]
    VecCls = SubprocVecEnv if subprocess and n_envs > 1 else DummyVecEnv
    venv = VecCls(factories)
    if normalize:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return venv


def make_ppo(venv, tensorboard_log: str | None = "outputs/tb", **kwargs):
    """PPO tuned for this continuous-control task."""
    defaults = dict(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.003,        # keep exploring; don't collapse to the lazy swing
        clip_range=0.2,
        use_sde=True,          # gSDE: temporally-correlated exploration (suits muscles)
        sde_sample_freq=4,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
    )
    defaults.update(kwargs)
    return PPO("MlpPolicy", venv, tensorboard_log=tensorboard_log, **defaults)


class EvalVideoCallback(BaseCallback):
    """Every `every` steps, render the current policy and save an mp4.

    Lets you scrub outputs/videos/ during/after training and see the swing emerge.
    """

    def __init__(self, body: BodyParams | None = None, every: int = 50_000,
                 out_dir: str = "outputs/videos", verbose: int = 0):
        super().__init__(verbose)
        self.body = body
        self.every = every
        self.out_dir = out_dir
        self._last = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.every:
            self._last = self.num_timesteps
            self._record()
        return True

    def _record(self):
        env = SwingEnv(body=self.body, render_mode="rgb_array")
        frames, _, summary = rollout(env, policy=self.model, deterministic=True)
        env.close()
        path = os.path.join(self.out_dir, f"swing_{self.num_timesteps:08d}.mp4")
        save_mp4(frames, path, fps=50)
        if self.verbose:
            print(f"[eval @ {self.num_timesteps}] "
                  f"impact={summary['impact_speed_mph']:.0f} mph -> {path}")
