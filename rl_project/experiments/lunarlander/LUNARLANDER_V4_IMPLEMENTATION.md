# LunarLander-v4 Implementation Notes

## Goal
Add a Lunar Lander environment variant that allows explicit control over the two main stochasticity sources:

- `initial_random_strength` (reset-time random impulse)
- `dispersion_strength` (engine dispersion noise during `step`)

This enables fully deterministic dynamics when both are `0.0`, with wind effects explicitly neutralized.

## What Was Implemented

### 1. New Tunable Environment Class
File: `rl_project/experiments/lunarlander/tunable_lunarlander.py`

- Added `TunableLunarLander`, a subclass of Gymnasium's `LunarLander`.
- Added constructor parameters:
  - `initial_random_strength: float` (default: upstream `INITIAL_RANDOM`)
  - `dispersion_strength: float` (default: upstream `1.0 / SCALE`)
  - `main_engine_power: float` (default: upstream `MAIN_ENGINE_POWER`)
  - `side_engine_power: float` (default: upstream `SIDE_ENGINE_POWER`)
  - `leg_spring_torque: float` (default: upstream `LEG_SPRING_TORQUE`)
  - `lander_mass_scale: float` (default: `1.0`)
  - `leg_mass_scale: float` (default: `1.0`)
  - `linear_damping: float | None` (default: `None`, i.e. upstream default)
  - `angular_damping: float | None` (default: `None`, i.e. upstream default)
  - `deterministic: bool` convenience flag
- When `deterministic=True`:
  - `initial_random_strength = 0.0`
  - `dispersion_strength = 0.0`
  - `enable_wind = False`
  - `wind_power = 0.0`
  - `turbulence_power = 0.0`
- Added input validation so both strengths must be `>= 0`.

This extra zeroing of wind/turbulence magnitudes is a defensive safeguard: even if
`enable_wind` is accidentally toggled later, wind/turbulence still remain neutral.

### 2. Reset-Time Randomness Control

- `reset()` temporarily overrides `gymnasium.envs.box2d.lunar_lander.INITIAL_RANDOM` with `self.initial_random_strength`.
- After calling `super().reset(...)`, the original constant is restored in a `finally` block.

### 3. Step-Time Dispersion Control

- Added `_DispersionRNGProxy`, which wraps `self.np_random`.
- The proxy rescales scalar `uniform(-1.0, 1.0)` calls used by LunarLander’s dispersion logic.
- Effective dispersion magnitude becomes `dispersion_strength`.
- The environment re-wraps RNG after reset because Gymnasium may replace `np_random` on seeding.

### 3b. Engine/Body Dynamics Control

- `step()` temporarily overrides upstream module constants:
  - `MAIN_ENGINE_POWER`
  - `SIDE_ENGINE_POWER`
- `reset()` temporarily overrides:
  - `LEG_SPRING_TORQUE`
- After reset, body-level properties are applied directly:
  - `lander_mass_scale` and `leg_mass_scale` (via Box2D body mass updates)
  - `linear_damping` and `angular_damping` (if provided)

### 4. Registration as `LunarLander-v4`

- Added:
  - `LUNARLANDER_V4_ID = "LunarLander-v4"`
  - `ensure_lunarlander_v4_registered()`
- Registration entry point:
  - `rl_project.experiments.lunarlander.tunable_lunarlander:TunableLunarLander`
- Registration is idempotent:
  - If `LunarLander-v4` is already registered with the same entry point, it is left unchanged.
  - If registered differently, it is replaced.
- Module calls `ensure_lunarlander_v4_registered()` at import time.

## Integration Into Existing Training/Eval Pipeline
File: `rl_project/experiments/lunarlander/train_source_policy.py`

- `_resolve_lunarlander_dynamics(...)` now reads and validates:
  - `initial_random_strength`
  - `dispersion_strength`
- `_load_task_settings(...)` now forwards these fields from YAML settings.
- `_make_lunarlander_env(...)` now:
  - Calls `ensure_lunarlander_v4_registered()` before `gym.make(...)`.
  - Passes `initial_random_strength` and `dispersion_strength` into env kwargs when provided.
- The new fields are now included in:
  - environment creation kwargs
  - printed runtime configuration
  - saved `run_summary.yaml` metadata

File: `rl_project/experiments/lunarlander/rollout_policy_video.py`

- Added support for reading/writing these dynamics fields from run metadata.
- When reconstructing env config for rollout, these values are passed to `_make_lunarlander_env(...)`.

File: `rl_project/experiments/lunarlander/settings/task_settings.yaml`

- Added nullable keys under each role:
  - `initial_random_strength: null`
  - `dispersion_strength: null`

## Usage

### Python
```python
import gymnasium as gym
from rl_project.experiments.lunarlander.tunable_lunarlander import ensure_lunarlander_v4_registered

ensure_lunarlander_v4_registered()
env = gym.make(
    "LunarLander-v4",
    continuous=False,
    enable_wind=False,
    initial_random_strength=0.0,
    dispersion_strength=0.0,
)
```

### Task settings YAML
```yaml
my_setting:
  env_id: LunarLander-v4
  continuous: false
  append_task_id: true
  source:
    task_id: 0.0
    gravity: null
    enable_wind: false
    wind_power: null
    turbulence_power: null
    initial_random_strength: 0.0
    dispersion_strength: 0.0
    action_repeat: 1
    action_delay: 0
    action_noise_prob: 0.0
    action_noise_mode: noop
    mark_out_of_viewport_as_unsafe: false
```

## Notes and Limitations

- The dispersion proxy intentionally targets the current upstream pattern (`uniform(-1, 1)` scalar calls used for dispersion).
- If upstream Gymnasium internals change, this hook may need to be updated.
- Deterministic policy rollout also requires deterministic action selection (`argmax`) in evaluation scripts.
