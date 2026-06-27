"""Body parameterization and MuJoCo model (MJCF) generation.

The whole point of this project is that body proportions change the swing. So the
physics model is *generated* from a small set of body parameters. Phase 1 gets one
body swinging; Phase 2 sweeps these parameters and compares the learned swings.

Phase 1 model is a planar (x-z) kinetic chain — a triple pendulum:

    shoulder hinge -> upper arm -> elbow hinge -> forearm -> wrist hinge -> club

All three hinges rotate about the y axis, so the whole motion lives in the vertical
x-z plane. Splitting the lead arm into an upper arm + forearm at an elbow lets the
agent create lag and store/release energy through the chain the way a real swing
does, and exposes upper-arm vs forearm proportions as body-type parameters.

Convention:
    x  = down-the-line / target direction (ball flies toward +x)
    z  = up
    y  = out of the swing plane (rotation axis for all hinges)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class BodyParams:
    """Segment lengths and masses describing one golfer + club.

    Lengths are in meters, masses in kilograms. Defaults are a roughly average
    adult male with a standard driver. Vary these to make "body types" later.
    """

    upper_arm_length: float = 0.32   # shoulder -> elbow, m
    forearm_length: float = 0.30     # elbow -> wrist, m
    club_length: float = 1.12        # wrist -> clubhead (driver), m

    upper_arm_mass: float = 2.2      # kg
    forearm_mass: float = 1.8        # kg (forearm + hand)
    club_mass: float = 0.31          # shaft + grip mass, kg
    clubhead_mass: float = 0.20      # concentrated mass at the clubhead, kg

    shoulder_height: float | None = None  # height of shoulder pivot; auto if None

    # lower body: a pelvis that shifts laterally toward the target, and a torso/spine
    # that tilts in the swing plane. Together they move the shoulder pivot — which is
    # what lets the low point move ahead of the ball (shaft lean) and adds the
    # proximal links of the kinetic chain.
    hip_height: float = 1.0          # pelvis pivot height, m
    pelvis_mass: float = 6.0         # kg
    torso_mass: float = 20.0         # kg (trunk)
    pelvis_shift_lo: float = -0.15   # how far the pelvis can shift back (−x), m
    pelvis_shift_hi: float = 0.40    # how far it can shift toward the target (+x), m
    spine_tilt_range: float = 0.6    # spine tilt limit, ± rad

    max_shoulder_torque: float = 250.0    # N·m at the shoulder (big proximal muscles)
    max_elbow_torque: float = 120.0       # N·m at the elbow
    max_wrist_torque: float = 40.0        # N·m at the wrist (the "release")
    max_spine_torque: float = 300.0       # N·m trunk (side bend / tilt)
    max_pelvis_force: float = 1000.0      # N lateral drive (lower body, abstracted)

    @property
    def spine_length(self) -> float:
        """Pelvis -> shoulder distance (when the spine is vertical)."""
        return self.resolved_shoulder_height() - self.hip_height

    @property
    def arm_length(self) -> float:
        """Total lead-arm length, shoulder -> wrist."""
        return self.upper_arm_length + self.forearm_length

    @property
    def reach(self) -> float:
        """Full reach, shoulder -> clubhead at full extension."""
        return self.arm_length + self.club_length

    def resolved_shoulder_height(self) -> float:
        """Shoulder pivot height so the bottom of the clubhead arc sits ~ground.

        Bottom of arc is at shoulder_height - reach. We want a small positive
        clearance so the ball can sit on the ground at impact.
        """
        if self.shoulder_height is not None:
            return self.shoulder_height
        return self.reach + 0.05

    def ball_position(self) -> tuple[float, float]:
        """(x, z) of the ball: at the bottom of the clubhead arc, below the shoulder."""
        h = self.resolved_shoulder_height()
        return (0.0, h - self.reach)

    def as_dict(self) -> dict:
        return asdict(self)


def _muscle_pair(joint: str, jrange: tuple[float, float], force: float) -> str:
    """An antagonist Hill-type muscle pair on one hinge (the actuation seam).

    Two joint-attached MuJoCo muscles with opposite gear so the joint can be driven
    both ways. `force` is the peak isometric force (≈ peak joint torque at gear 1).
    Muscles carry built-in force–length, force–velocity, and activation dynamics, so
    the speed cap is physical. `ctrl` in [0,1] is excitation; `data.act` is the
    activation state used for the Σ(activation²) effort term.
    """
    lo, hi = jrange
    lr_pos = f"{lo} {hi}"                 # length range for gear +1 (length = +qpos)
    lr_neg = f"{-hi} {-lo}"               # length range for gear -1 (length = -qpos)
    return (
        f'<muscle name="{joint}_flex" joint="{joint}" gear="1" '
        f'ctrlrange="0 1" force="{force}" lengthrange="{lr_pos}"/>\n    '
        f'<muscle name="{joint}_ext" joint="{joint}" gear="-1" '
        f'ctrlrange="0 1" force="{force}" lengthrange="{lr_neg}"/>'
    )


def _actuator_block(p: "BodyParams", actuation: str) -> str:
    """Build the <actuator> contents for either muscles (default) or torque motors.

    Torque is kept behind the same seam for quick debugging / ablation; the brief's
    biology cap requires muscles, so that's the default.
    """
    if actuation == "muscle":
        return "\n    ".join([
            _muscle_pair("pelvis_shift", (p.pelvis_shift_lo, p.pelvis_shift_hi), p.max_pelvis_force),
            _muscle_pair("spine_tilt", (-p.spine_tilt_range, p.spine_tilt_range), p.max_spine_torque),
            _muscle_pair("shoulder", (-3.4, 3.4), p.max_shoulder_torque),
            _muscle_pair("elbow", (-0.1, 2.6), p.max_elbow_torque),
            _muscle_pair("wrist", (-2.0, 2.0), p.max_wrist_torque),
        ])
    elif actuation == "torque":
        return "\n    ".join([
            f'<motor name="pelvis_shift" joint="pelvis_shift" gear="1" ctrlrange="-{p.max_pelvis_force} {p.max_pelvis_force}"/>',
            f'<motor name="spine_tilt" joint="spine_tilt" gear="1" ctrlrange="-{p.max_spine_torque} {p.max_spine_torque}"/>',
            f'<motor name="shoulder" joint="shoulder" gear="1" ctrlrange="-{p.max_shoulder_torque} {p.max_shoulder_torque}"/>',
            f'<motor name="elbow" joint="elbow" gear="1" ctrlrange="-{p.max_elbow_torque} {p.max_elbow_torque}"/>',
            f'<motor name="wrist" joint="wrist" gear="1" ctrlrange="-{p.max_wrist_torque} {p.max_wrist_torque}"/>',
        ])
    raise ValueError(f"unknown actuation: {actuation!r} (use 'muscle' or 'torque')")


def build_mjcf(p: BodyParams, actuation: str = "muscle") -> str:
    """Return a MuJoCo MJCF XML string for the given body parameters.

    actuation: "muscle" (Hill-type antagonist pairs, the real model) or "torque"
    (plain motors, for debugging/ablation). Both sit behind the same seam.
    """
    h = p.resolved_shoulder_height()
    uarm_r = 0.045         # upper-arm capsule radius
    farm_r = 0.038         # forearm capsule radius
    club_r = 0.012         # shaft radius
    head_r = 0.050         # clubhead sphere radius (also the collision radius);
                           # ~10 cm driver head — big enough that a fast clubhead
                           # can't tunnel through the ball in one timestep
    ball_r = 0.0213        # golf ball radius, m
    ball_m = 0.0459        # golf ball mass, kg
    ball_x, strike_z = p.ball_position()   # clubhead-pass height at the bottom of the arc
    tee_top = strike_z - ball_r            # tee holds the ball so its center sits at strike_z

    # Contact bitmasks (two geoms collide iff contype_i & conaffinity_j is nonzero,
    # in either order). Layout: bit1=ground/tee, bit2=ball, bit4=clubface.
    #   ball<->ground/tee: yes    clubface<->ball: yes    clubface<->ground: NO
    return f"""<mujoco model="golf_triple_pendulum">
  <compiler angle="radian" autolimits="true"/>
  <!-- small timestep so a fast clubhead doesn't tunnel through the ball at impact;
       0.001 balances contact fidelity against training throughput -->
  <option timestep="0.001" gravity="0 0 -9.81" integrator="RK4"/>

  <default>
    <joint damping="0.05" armature="0.01"/>
    <geom contype="0" conaffinity="0"/>
  </default>

  <worldbody>
    <light pos="0 -2 4" dir="0 0.4 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0" rgba="0.3 0.4 0.3 1"
          contype="1" conaffinity="2"/>

    <!-- fixed camera looking down the -y axis at the swing plane -->
    <camera name="swing" pos="0.2 -4.2 {h * 0.55:.3f}" xyaxes="1 0 0 0 0 1"/>

    <!-- address reference (where the clubhead should arrive, fast, at impact) -->
    <site name="address" pos="{ball_x} 0 {strike_z}" size="0.01" rgba="0 1 1 0.4"/>

    <!-- tee holding the ball at clubhead-pass height -->
    <geom name="tee" type="capsule" fromto="{ball_x} 0 0  {ball_x} 0 {tee_top}"
          size="0.006" rgba="0.9 0.9 0.2 1" contype="1" conaffinity="2"/>

    <!-- the ball: a free body that flies when struck -->
    <body name="ball" pos="{ball_x} 0 {strike_z}">
      <freejoint name="ball_free"/>
      <geom name="ball" type="sphere" size="{ball_r}" mass="{ball_m}"
            rgba="1 1 1 1" contype="2" conaffinity="7"
            solref="0.002 1" friction="0.3 0.01 0.001"/>
    </body>

    <!-- pelvis: shifts laterally toward the target (lower-body drive, abstracted) -->
    <body name="pelvis" pos="0 0 {p.hip_height}">
      <joint name="pelvis_shift" type="slide" axis="1 0 0"
             range="{p.pelvis_shift_lo} {p.pelvis_shift_hi}" damping="8"/>
      <geom name="pelvis" type="box" size="0.12 0.12 0.08"
            mass="{p.pelvis_mass}" rgba="0.4 0.5 0.7 1"/>

      <!-- torso / spine: tilts in the swing plane (side bend / forward lean). The
           spine is an inverted pendulum, so it carries passive postural stiffness
           (tone/ligaments) that holds it upright at address; the spine muscles
           (300 N·m) easily overpower it to tilt during the swing. -->
      <body name="torso" pos="0 0 0">
        <joint name="spine_tilt" type="hinge" axis="0 1 0"
               range="-{p.spine_tilt_range} {p.spine_tilt_range}"
               damping="3" stiffness="250"/>
        <geom name="torso" type="capsule" fromto="0 0 0  0 0 {p.spine_length}"
              size="0.07" mass="{p.torso_mass}" rgba="0.5 0.6 0.8 1"/>

        <!-- shoulder pivot at the top of the spine -->
        <body name="upper_arm" pos="0 0 {p.spine_length}">
          <joint name="shoulder" type="hinge" axis="0 1 0" range="-3.4 3.4"/>
          <geom name="upper_arm" type="capsule" fromto="0 0 0  0 0 -{p.upper_arm_length}"
                size="{uarm_r}" mass="{p.upper_arm_mass}" rgba="0.85 0.7 0.55 1"/>

          <!-- elbow pivot at the end of the upper arm -->
          <body name="forearm" pos="0 0 -{p.upper_arm_length}">
            <joint name="elbow" type="hinge" axis="0 1 0" range="-0.1 2.6"/>
            <geom name="forearm" type="capsule" fromto="0 0 0  0 0 -{p.forearm_length}"
                  size="{farm_r}" mass="{p.forearm_mass}" rgba="0.8 0.62 0.47 1"/>

            <!-- wrist pivot at the end of the forearm -->
            <body name="club" pos="0 0 -{p.forearm_length}">
              <joint name="wrist" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
              <geom name="shaft" type="capsule" fromto="0 0 0  0 0 -{p.club_length}"
                    size="{club_r}" mass="{p.club_mass}" rgba="0.2 0.2 0.2 1"/>
              <!-- no physics contact on the clubhead: impact is handled by an analytic
                   off-center collision model in the env (consistent smash, no tunneling).
                   The ball is launched kinematically for visualization. -->
              <geom name="clubhead" type="sphere" pos="0 0 -{p.club_length}"
                    size="{head_r}" mass="{p.clubhead_mass}" rgba="0.8 0.1 0.1 1"
                    contype="0" conaffinity="0"/>
              <site name="clubhead" pos="0 0 -{p.club_length}" size="0.01" rgba="1 0 0 1"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    {_actuator_block(p, actuation)}
  </actuator>
</mujoco>
"""


if __name__ == "__main__":
    p = BodyParams()
    print(f"arm length (upper+fore): {p.arm_length:.3f} m   reach: {p.reach:.3f} m")
    print(f"shoulder height: {p.resolved_shoulder_height():.3f} m")
    print(f"ball position (x,z): {p.ball_position()}")
