Implement simple safety-shield synthesis for the discrete-action Gymnasium environments `MountainCar-v0`, `CartPole-v1`, and `LunarLander-v3`.

The purpose of the shield is to expose, for every continuous state \(s\), a set of admissible actions

$$
\mathcal A_{\mathrm{safe}}(s)\subseteq\mathcal A
$$

such that actions that would obviously move the system further toward a safety violation are filtered out. The implementation should be simple, transparent, deterministic, and easy to use during RL training. It does not need to provide formal guarantees beyond the explicitly defined rule-based safety specification.

### General requirements

Create a common shield interface, for example:

```python
class SafetyShield:
    def get_safe_actions(self, state) -> list[int]:
        ...

    def is_safe_action(self, state, action: int) -> bool:
        ...

    def shield_action(self, state, proposed_action: int) -> int:
        ...
```

`get_safe_actions(state)` should return all admissible discrete actions.

`shield_action(state, proposed_action)` should:

1. return the proposed action unchanged if it is safe;
2. otherwise replace it with a safe action according to a deterministic fallback rule;
3. clearly define what happens if the safe-action set is empty.

Prefer returning the full admissible action set, because later this will be used as supervision of the form

$$
(s,\mathcal A_{\mathrm{safe}}(s))
$$

for safe policy optimisation.

Keep all thresholds configurable through constructor arguments or a dataclass. Do not hard-code experiment-specific constants throughout the implementation.

Add unit tests and a small visualization/demo script for each environment showing when the shield activates and which actions are allowed.

---

## 1. MountainCar-v0

The state is

$$
s=(x,v),
$$

where \(x\) is position and \(v\) is velocity.

The action space is:

```text
0 = accelerate left
1 = no acceleration
2 = accelerate right
```

Define configurable safety margins near the left and right position boundaries.

The intended qualitative safety rule is:

* near the left boundary, disallow actions that push the car further left;
* near the right boundary, disallow actions that push the car further right;
* away from the safety-critical boundary regions, all actions are safe.

However, because MountainCar has momentum, do not implement this purely as:

```python
if near_left:
    disallow action 0
```

if a more accurate one-step transition can be computed easily.

Prefer implementing the shield using the known MountainCar dynamics:

$$
v' = \operatorname{clip}
\left(
v + 0.001(a-1) - 0.0025\cos(3x),
-v_{\max},v_{\max}
\right),
$$

$$
x' = \operatorname{clip}(x+v',x_{\min},x_{\max}).
$$

Define a configurable safe positional set such as

$$
x_{\mathrm{safe,min}}\le x\le x_{\mathrm{safe,max}}.
$$

Then define

$$
\mathcal A_{\mathrm{safe}}(s)
=
\{a:x'(s,a)\text{ does not move the system farther outside/toward violation of the safety margin}\}.
$$

A reasonable implementation is to prefer actions whose predicted successor stays within the safe interval, and when already inside a safety-critical margin, reject actions that worsen the relevant boundary violation.

Document the exact rule used.

---

## 2. CartPole-v1

The state is

$$
s=(x,\dot x,\theta,\dot\theta).
$$

The action space is:

```text
0 = push cart left
1 = push cart right
```

Implement safety specifications based on configurable margins for:

1. cart position:

   $$
   |x|\le x_{\mathrm{safe}}
   $$

2. pole angle:

   $$
   |\theta|\le \theta_{\mathrm{safe}}.
   $$

The shield should become active before Gymnasium's actual termination thresholds.

Prefer using the one-step CartPole dynamics to evaluate both actions and construct

$$
\mathcal A_{\mathrm{safe}}(s)
=
\left\{
a:
s'(s,a)\text{ improves or preserves safety}
\right\}.
$$

At minimum:

* near the right cart boundary, actions that push the cart further right should be disallowed;
* near the left cart boundary, actions that push the cart further left should be disallowed;
* when the pole angle is close to its safety threshold, prefer/disallow actions according to whether the predicted next pole angle moves toward or away from the upright region.

Because cart-position safety and pole-angle safety may conflict, implement an explicit combination rule.

A good default is:

1. compute the one-step successor for every action;
2. assign each successor a safety-violation score such as

$$
V(s')
=
w_x\max(0,|x'|-x_{\mathrm{safe}})
+
w_\theta\max(0,|\theta'|-\theta_{\mathrm{safe}});
$$

3. define actions with zero predicted violation as safe;
4. if no action has zero violation, retain actions minimizing the violation score.

Alternatively, use a simpler logically equivalent rule if it is easier to interpret, but document it clearly.

The shield must never return an empty set unless explicitly configured to do so.

---

## 3. LunarLander-v3

Use the discrete-action version.

The state includes approximately:

$$
(x,y,v_x,v_y,\theta,\dot\theta,\text{left-contact},\text{right-contact}).
$$

The actions are:

```text
0 = do nothing
1 = fire left orientation engine
2 = fire main engine
3 = fire right orientation engine
```

Implement a simple, configurable safety specification based on three components.

### A. Horizontal safety

Define a safe horizontal operating corridor:

$$
|x|\le x_{\mathrm{safe}}.
$$

Near the right boundary, disallow actions expected to make horizontal motion worse.

Near the left boundary, do the symmetric operation.

Because the exact effect of the orientation engines depends on lander orientation, preferably use either:

* a one-step cloned environment simulation, or
* a documented approximate directional rule.

If Gymnasium/Box2D state cloning is cumbersome, it is acceptable to use transparent heuristic rules based on \(x\), \(v_x\), and \(\theta\).

### B. Vertical-speed safety near the ground

Define configurable thresholds:

$$
y<h_{\mathrm{critical}}
$$

and

$$
v_y<v_{\mathrm{safe,min}}.
$$

When the lander is low and descending too quickly, require or strongly prefer the main engine action.

For example, under sufficiently unsafe descent:

```text
safe actions = {2}
```

unless there is a strong technical reason to include another action.

### C. Attitude safety near landing

Define

$$
|\theta|\le \theta_{\mathrm{safe}}.
$$

At low altitude, disallow orientation-engine actions that are expected to increase \(|\theta|\), and allow actions that correct the tilt.

Combine the three safety components through intersection where possible:

$$
\mathcal A_{\mathrm{safe}}(s)
=
\mathcal A_x(s)
\cap
\mathcal A_v(s)
\cap
\mathcal A_\theta(s).
$$

If the intersection is empty, use a deterministic priority rule. The default priority should be:

1. prevent dangerous vertical descent;
2. prevent excessive attitude;
3. prevent horizontal-boundary violation.

Document this fallback explicitly.

---

## Architecture

Create something approximately like:

```text
shields/
    base.py
    mountain_car.py
    cartpole.py
    lunar_lander.py
    config.py
tests/
    test_mountain_car_shield.py
    test_cartpole_shield.py
    test_lunar_lander_shield.py
scripts/
    demo_shields.py
```

Adapt this structure to the existing repository conventions if appropriate.

Each shield should support:

```python
safe_actions = shield.get_safe_actions(obs)

safe = shield.is_safe_action(obs, action)

executed_action = shield.shield_action(obs, proposed_action)
```

Also expose diagnostic information if convenient, for example:

```python
safe_actions, info = shield.get_safe_actions(obs, return_info=True)
```

where `info` may contain:

```python
{
    "shield_active": True,
    "violated_constraints": [...],
    "predicted_next_states": {...},
    "reason": "...",
}
```

This diagnostic output would be useful for dataset generation and debugging.

---

## Dataset-generation support

Add a simple helper that can collect tuples

$$
(s_t,\mathcal A_{\mathrm{safe}}(s_t))
$$

during interaction.

For example:

```python
dataset.append({
    "state": obs.copy(),
    "safe_actions": safe_actions,
})
```

Do not couple this tightly to a particular RL library.

The goal is to later use these state/safe-action-set pairs for training or verification of a neural policy satisfying

$$
\arg\max_a z_{\theta,a}(s)
\in
\mathcal A_{\mathrm{safe}}(s).
$$

---

## Tests

Write unit tests covering at least:

### MountainCar

* central state → all actions safe;
* near left boundary → action pushing farther left is rejected where appropriate;
* near right boundary → action pushing farther right is rejected;
* momentum is taken into account.

### CartPole

* central/upright state → both actions safe where appropriate;
* near right cart boundary → rightward push rejected;
* near left cart boundary → leftward push rejected;
* large positive/negative pole angle → corrective action preferred;
* no empty safe-action set.

### LunarLander

* nominal high-altitude state → all or most actions safe;
* low altitude + excessive downward speed → main engine required;
* near left/right horizontal boundary → outward action rejected;
* excessive tilt near the ground → action worsening tilt rejected;
* conflicting constraints invoke the documented priority rule.

---

## Demo / validation

Create a script that runs each environment with either a random policy or a simple existing policy and reports:

* number of environment steps;
* number and percentage of states where the shield activated;
* number of proposed actions overridden;
* distribution of safe-action-set sizes;
* termination / episode outcome.

Optionally render a few episodes.

Keep the implementation minimal and research-friendly. Avoid introducing unnecessary framework dependencies.

Before modifying code, inspect the repository structure and reuse existing abstractions where sensible. After implementation, run the tests and provide a concise summary of:

1. files added/changed;
2. exact safety rules implemented for each environment;
3. default threshold values;
4. any approximations made, especially for LunarLander;
5. test results;
6. example usage.
