# rl-golf

Predictive RL golf swing. A parameterized upper-body skeleton driven by **Hill-type
muscles** learns an efficient swing **from scratch** (no reference motion), then
re-learns it under a **changed body** to study how the whole swing reorganizes
(clubhead speed, swing plane, X-factor, kinematic sequence). Train each body
cold-start; compare distributions across seeds.

## Status (brief build order)

1. ✅ Get running: torque-actuated planar model, one body (proven — PPO learns a swing).
2. 🔜 Biological realism: now on **Hill-type muscle actuators** (antagonist pairs,
   force–velocity cap + activation dynamics). Tune strengths for a realistic
   ~100–120 mph clubhead cap.
3. ⏳ Lock one validated nominal swing (realistic speed + proximal→distal sequence).
4. ⏳ Go 3D (pelvis + thorax) so X-factor / swing-plane / side-bend are measurable.
5. ⏳ Sweep body geometry, 3–5 seeds each, compare distributions.

Reward (impact only, no ball flight simulated):
`w_speed·clubhead_speed_at_impact − w_quality·centered_contact − w_effort·Σ(activation²)`,
with the effort weight **frozen across bodies**.

## Layout

```
src/rl_golf/
  body.py        BodyParams -> MuJoCo MJCF; actuation seam ("muscle" | "torque")
  swing_env.py   Gymnasium env: actuation-agnostic action, impact-only reward, Σact² effort
  train.py       PPO setup + EvalVideoCallback (records a clip as it learns)
  viz.py         rollout / render / filmstrip helpers
notebooks/
  train_swing.ipynb   build body, baselines, train PPO, watch, diagnostics
```

## Setup

```bash
conda create -n rl-golf python=3.12
conda activate rl-golf
pip install -e .
pip install jupyterlab ipykernel tensorboard
python -m ipykernel install --user --name rl-golf --display-name "Python 3 (rl-golf)"
```

## Run

Open `notebooks/train_swing.ipynb` (kernel: *Python 3 (rl-golf)*) and run the cells.
Live curves: `tensorboard --logdir outputs/tb`.
