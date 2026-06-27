"""Rendering / rollout helpers shared by the notebook, scripts, and live viewer.

Keep the heavy logic here so notebooks stay thin: they import these and just look
at the pictures.
"""

from __future__ import annotations

import numpy as np

from rl_golf.swing_env import SwingEnv


def rollout(env: SwingEnv, policy=None, deterministic: bool = True, seed: int = 0):
    """Run one episode. Return (frames, info_history, summary).

    policy: anything with .predict(obs) -> (action, _) (e.g. an SB3 model). If
    None, applies zero torque (a passive drop) so you can sanity-check the env.
    Requires env.render_mode == "rgb_array" to collect frames.
    """
    obs, _ = env.reset(seed=seed)
    frames, history = [], []
    done = False
    peak_speed = 0.0
    while not done:
        if policy is None:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        else:
            action, _ = policy.predict(obs, deterministic=deterministic)
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        peak_speed = max(peak_speed, info["clubhead_speed"])
        history.append(info)
        if env.render_mode == "rgb_array":
            frames.append(env.render())
    last = history[-1] if history else {}
    summary = {
        "peak_speed_ms": peak_speed,
        "peak_speed_mph": peak_speed * 2.237,
        "impact_speed_ms": env.impact_speed,
        "impact_speed_mph": env.impact_speed * 2.237,
        "impact_dist_m": env.impact_dist,
        "ball_speed_mph": last.get("ball_speed", 0.0) * 2.237,
        "smash_factor": last.get("smash_factor", 0.0),
        "took_backswing": last.get("took_backswing", False),
        "early_hit": last.get("early_hit", False),
        "total_effort": env.total_effort,
        "steps": len(history),
    }
    return frames, history, summary


def save_mp4(frames, path: str, fps: int = 50):
    import os
    import imageio.v2 as imageio

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)
    return path


def filmstrip(frames, n: int = 6):
    """Horizontally concatenate n evenly-spaced frames into one image array."""
    if not frames:
        return None
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    return np.concatenate([frames[i] for i in idx], axis=1)
