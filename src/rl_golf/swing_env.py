"""Gymnasium environment: learn an efficient golf swing, scored at impact.

Per the project brief this is **predictive, not imitation** — no reference motion.
The agent applies muscle excitations (or torques, behind the same seam) and is
rewarded only for what happens at impact:

    reward = w_speed * clubhead_speed_at_impact
           - w_quality * impact_quality_penalty      (2D: centered contact)
           - w_effort * Σ(activation²)                (frozen across bodies)

No ball flight / spin / aero is simulated or rewarded. A potential-based approach
shaping term is added purely for trainability (it telescopes, so it does not bias
the optimal swing and is not reference tracking).

Effort uses the muscle activation states (`data.act`); with torque actuators it
falls back to normalized command magnitude. The actuator is read generically so the
torque↔muscle swap needs no env changes.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from rl_golf.body import BodyParams, build_mjcf

JOINTS = ("pelvis_shift", "spine_tilt", "shoulder", "elbow", "wrist")

# Address starting pose: pelvis centered, spine vertical, arms + club hanging straight
# down so the clubhead starts AT the ball — like setting up to a real shot. The agent
# must take the club back (a real backswing) and deliver it back down through the ball.
ADDRESS = {"pelvis_shift": 0.0, "spine_tilt": 0.0, "shoulder": 0.0, "elbow": 0.0, "wrist": 0.0}


class SwingEnv(gym.Env):
    """One body, one club, one swing, scored at impact."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(
        self,
        body: BodyParams | None = None,
        actuation: str = "muscle",
        max_time: float = 1.2,
        frame_skip: int = 10,
        w_ball: float = 4.0,         # weight on ball speed — must DOMINATE the shaping
        cor: float = 0.83,           # coefficient of restitution (driver ~0.83 -> smash ~1.5)
        w_effort: float = 0.005,     # FROZEN across bodies (brief requirement)
        approach_reward: float = 1.0,  # faint guide only; kept small so it can't out-vote speed
        backswing_frac: float = 0.55,  # club must be taken back this fraction of reach
        early_hit_penalty: float = 20.0,  # penalty for hitting the ball with no backswing
        early_hit_vx: float = 1.0,        # +x clubhead speed near ball that counts as a shove
        init_noise: float = 0.05,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.body = body or BodyParams()
        self.actuation = actuation
        self.model = mujoco.MjModel.from_xml_string(
            build_mjcf(self.body, actuation=actuation))
        self.data = mujoco.MjData(self.model)

        self.frame_skip = frame_skip
        self.dt = self.model.opt.timestep * frame_skip
        self.max_steps = int(round(max_time / self.dt))
        self.w_ball = w_ball
        self.cor = cor
        self.w_effort = w_effort
        self.approach_reward = approach_reward
        self.early_hit_penalty = early_hit_penalty
        self.early_hit_vx = early_hit_vx
        self.init_noise = init_noise
        self.render_mode = render_mode

        # cached ids
        self._qadr = np.array([self.model.joint(j).qposadr[0] for j in JOINTS])
        self._dadr = np.array([self.model.joint(j).dofadr[0] for j in JOINTS])
        self._clubhead_id = self.model.site("clubhead").id
        self._target = np.array(self.body.ball_position())  # (x, z) strike point
        self._ball_vadr = self.model.joint("ball_free").dofadr[0]  # ball lin-vel dofs

        # analytic impact model parameters (off-center sphere-sphere collision)
        self._r_club = float(self.model.geom("clubhead").size[0])
        self._r_ball = float(self.model.geom("ball").size[0])
        self._R_sum = self._r_club + self._r_ball       # contact happens within this
        self._M = float(self.body.clubhead_mass)         # clubhead mass
        self._m = float(self.model.body("ball").mass[0]) # ball mass

        # a real backswing takes the clubhead BEHIND the ball (−x). It must reach at
        # least this far back to count (so a forward shove from address doesn't).
        self._backswing_x = -backswing_frac * self.body.reach

        # actuator command range, for mapping action [-1,1] -> ctrl
        self._ctrl_lo = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_hi = self.model.actuator_ctrlrange[:, 1].copy()
        self._nu = self.model.nu
        self._na = self.model.na  # number of activation states (muscles)

        self._step_count = 0
        self._reset_impact_trackers()

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self._nu,), dtype=np.float32)
        obs_dim = self._obs().shape[0]
        high = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self._renderer = None

    # ------------------------------------------------------------------ helpers
    def _reset_impact_trackers(self):
        self.impact_dist = np.inf       # closest clubhead approach AFTER the backswing
        self.impact_speed = 0.0         # clubhead speed at that closest approach
        self.peak_clubhead_speed = 0.0
        self.total_effort = 0.0
        self._ball_kicked = False        # cosmetic launch fired yet?
        self.took_backswing = False      # has the club been taken back behind the ball?
        self.min_clubhead_x = 0.0        # farthest BACK (−x) the clubhead reached
        self.early_hit = False           # did it shove the ball with no backswing?

    def ball_launch_speed(self) -> float:
        """Analytic ball launch speed from the best impact this swing.

        Off-center sphere-sphere collision: a centered head-on hit transfers
        (1+e)·M/(M+m) of clubhead speed (smash ≈ 1.5); a glancing hit at miss
        distance d tapers by √(1−(d/R)²) to 0 at the edge. Consistent at all speeds,
        no contact-sim artifacts.
        """
        if self.impact_dist >= self._R_sum:
            return 0.0
        centered = (1.0 - (self.impact_dist / self._R_sum) ** 2) ** 0.5
        return (1.0 + self.cor) * self._M / (self._M + self._m) * self.impact_speed * centered

    def smash_factor(self) -> float:
        return (self.ball_launch_speed() / self.impact_speed
                if self.impact_speed > 1e-6 else 0.0)

    def _clubhead_state(self):
        pos = self.data.site_xpos[self._clubhead_id][[0, 2]]
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_SITE,
            self._clubhead_id, vel6, False)
        vel = vel6[3:6][[0, 2]]
        return pos, vel

    def _effort(self, action) -> float:
        """Σ(activation²): muscle activation states, or normalized torque cmd."""
        if self._na > 0:
            return float(np.sum(self.data.act ** 2))
        return float(np.sum(np.clip(action, -1, 1) ** 2))

    def _obs(self) -> np.ndarray:
        q = self.data.qpos[self._qadr]
        qd = self.data.qvel[self._dadr]
        pos, vel = self._clubhead_state()
        phase = np.array([self._step_count / self.max_steps], dtype=np.float64)
        parts = [np.sin(q), np.cos(q), qd * 0.1, pos, vel * 0.05, phase]
        if self._na > 0:                      # muscle activation states
            parts.append(self.data.act.copy())
        return np.concatenate(parts).astype(np.float32)

    # -------------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        for j in JOINTS:
            adr = self.model.joint(j).qposadr[0]
            self.data.qpos[adr] = ADDRESS[j] + self.np_random.uniform(
                -self.init_noise, self.init_noise)
        self._step_count = 0
        self._reset_impact_trackers()
        mujoco.mj_forward(self.model, self.data)
        pos, _ = self._clubhead_state()
        self._prev_dist = float(np.linalg.norm(pos - self._target))
        self._prev_x = float(pos[0])
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        # map [-1,1] -> actuator ctrlrange (muscles: ->[0,1]; torque: ->[-T,T])
        self.data.ctrl[:] = self._ctrl_lo + (action + 1.0) * 0.5 * (
            self._ctrl_hi - self._ctrl_lo)
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self._step_count += 1

        pos, vel = self._clubhead_state()
        x = float(pos[0])   # clubhead x: −x is behind the ball (backswing side)
        z = float(pos[1])
        vx = float(vel[0])  # +x = moving toward the target (a real downswing strike)
        dist = float(np.linalg.norm(pos - self._target))
        speed = float(np.linalg.norm(vel))
        self.peak_clubhead_speed = max(self.peak_clubhead_speed, speed)

        reward = 0.0

        # --- backswing gate: the club must be taken back (behind the ball) first ---
        self.min_clubhead_x = min(self.min_clubhead_x, x)
        if self.min_clubhead_x <= self._backswing_x:
            self.took_backswing = True

        # clubhead reaching the front (target) side of the ball at ball height. If this
        # happens BEFORE a backswing, the club was shoved forward, not swung. Threshold
        # (0.1 m) clears the address noise and can't be skipped between steps.
        on_front_side = x > 0.10 and abs(z - self._target[1]) < 0.30

        if not self.took_backswing:
            # phase 1 (takeaway): reward taking the club BACK (−x)...
            reward += self.approach_reward * (self._prev_x - x)
            # ...and penalize shoving to the front of the ball with no backswing
            if on_front_side and not self.early_hit:
                reward -= self.early_hit_penalty
                self.early_hit = True
        else:
            # phase 2 (downswing): reward bringing the club back toward the ball
            reward += self.approach_reward * (self._prev_dist - dist)
            # impact = closest approach while the clubhead is moving TOWARD the target
            # (vx > 0). A backswing pass (clubhead moving −x) is not a strike and is
            # ignored, so the agent can't farm speed by hitting on the way back.
            if vx > 0.0 and dist < self.impact_dist:
                self.impact_dist = dist
                self.impact_speed = speed
            # cosmetic: launch the (kinematic) ball along clubhead travel once it
            # has passed the ball, so the visual matches the metric (not a contact)
            if (not self._ball_kicked and self.impact_dist < self._R_sum
                    and dist > self.impact_dist + 1e-4):
                bs = self.ball_launch_speed()
                vdir = vel / (np.linalg.norm(vel) + 1e-9)
                self.data.qvel[self._ball_vadr:self._ball_vadr + 3] = [
                    vdir[0] * bs, 0.0, vdir[1] * bs]
                self._ball_kicked = True
        self._prev_dist = dist
        self._prev_x = x

        # effort, accumulated every step (frozen weight across bodies)
        effort = self._effort(action)
        self.total_effort += effort
        reward -= self.w_effort * effort

        terminated = False
        truncated = self._step_count >= self.max_steps
        # impact reward, paid once at episode end: analytic BALL launch speed (bundles
        # clubhead speed AND centered contact). Voided if the ball was shoved without a
        # real backswing. Clubhead speed stays the reported headline DV.
        ball_speed = self.ball_launch_speed()
        valid_hit = self.took_backswing and not self.early_hit
        if truncated and valid_hit:
            reward += self.w_ball * ball_speed

        info = {
            "clubhead_speed": speed,
            "impact_speed": self.impact_speed,
            "impact_dist": self.impact_dist,
            "ball_speed": ball_speed if valid_hit else 0.0,
            "smash_factor": self.smash_factor() if valid_hit else 0.0,
            "took_backswing": self.took_backswing,
            "early_hit": self.early_hit,
            "peak_clubhead_speed": self.peak_clubhead_speed,
            "total_effort": self.total_effort,
        }
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        cam = self.model.camera("swing").id
        self._renderer.update_scene(self.data, camera=cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
