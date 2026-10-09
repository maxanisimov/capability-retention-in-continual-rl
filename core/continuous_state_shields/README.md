# Continuous-state safety shields

This package provides deterministic, rule-based action filtering for the discrete
versions of `MountainCar-v0`, `CartPole-v1`, and `LunarLander-v3`. It depends only
on NumPy; Gymnasium is needed by the demo and dynamics-comparison tests.

## Rules and defaults

| Environment | Default rule | Default safety thresholds |
|---|---|---|
| MountainCar | When position is in `[-1.2, -1.0]` and velocity is negative, admit only push right (action 2). Otherwise admit every action. | unsafe event: contact with the physical left boundary (`x = -1.2`); safety-critical states: `x in [-1.2, -1.0]` and `v < 0` |
| CartPole | Apply Gymnasium's equations for two Euler steps. Keep every zero-violation action; if none exists, minimize `1.0 * position_violation + 5.0 * angle_violation`. | `|x| <= 2.2`; `|theta| <= 10 degrees`; two-step lookahead |
| LunarLander | Intersect active vertical, attitude, and horizontal heuristic sets. If empty, use vertical > attitude > horizontal priority. | `|x| <= 0.8`; dangerous descent below `y=0.5` and `vy < -0.4`; attitude limit `20 degrees` below `y=0.75` |

The CartPole margins precede its `2.4` position and `12` degree termination
limits. Two lookahead steps are used because, under Gymnasium's default Euler
integrator, the first position and angle update uses the old velocities and is
therefore action-independent.

LunarLander does not clone Box2D. Its documented approximation treats action 1
(left orientation engine) as positive horizontal thrust and negative angular
acceleration, action 3 as the reverse, and the main engine's horizontal effect as
proportional to `-sin(theta)`. An active component admits actions whose projected
absolute error is no worse than doing nothing. Dangerous low descent requires
action 2, including when it conflicts with the other components.

All constants are fields of the corresponding frozen config dataclass.

## Usage

```python
from continuous_state_shields import MountainCarShield, ShieldDatasetCollector

shield = MountainCarShield()
safe_actions, info = shield.get_safe_actions(observation, return_info=True)
executed_action = shield.shield_action(observation, proposed_action)

collector = ShieldDatasetCollector(shield)
collector.append(observation)
```

### Synthesized MountainCar reach-avoid shield

The dynamics-aware shield is stored as a compressed finite-grid artifact and
loaded explicitly:

```python
from continuous_state_shields import MountainCarReachAvoidShield

shield = MountainCarReachAvoidShield.load("mountaincar_reach_avoid_shield.npz")
safe_actions = shield.get_safe_actions(observation)
executed_action = shield.shield_action(observation, proposed_action)
```

Run `projects/safe_policy_optimisation/archive/stages/synthesise_mountaincar_reach_avoid_shield.py`
to regenerate the viability kernel, safe goal-reachable set, goal ranks, and
state-action mask from the exact MountainCar transition equations.

The default replacement for an unsafe proposal is the lowest-numbered safe
action. Pass `fallback_rule="highest"` or a deterministic callable to change it.
The supplied shields never produce an empty set. The base interface raises if an
implementation unexpectedly does; a deliberately empty custom shield may opt in,
but `shield_action` still raises because no truthful safe replacement exists.

Run the demo from the repository root after installing the package:

```bash
python core/scripts/demo_continuous_state_shields.py --env all --episodes 3
python core/scripts/demo_continuous_state_shields.py --env cartpole --render --show-activations
```
