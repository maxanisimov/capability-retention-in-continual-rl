# Poisoned Apple Environment

A Gymnasium-based grid world environment where an agent must collect safe apples while avoiding poisoned ones. This environment is designed for continual learning research.

## Features

- **Grid-based Navigation**: Agent moves in a grid world using discrete actions (UP, RIGHT, DOWN, LEFT)
- **Apple Collection**: Safe apples provide positive rewards, poisoned apples provide negative rewards
- **Configurable Setup**: Support for both random and fixed positioning
- **Episode Limits**: Maximum steps equal to the number of apples in the environment
- **Multiple Rendering Modes**: Human-readable console output or RGB array for visualization

## Installation

```python
from experiments.poisoned_apple import PoisonedAppleEnv, make_task1_env, make_task2_env
```

## Usage

### Random Positioning (Default)

```python
from experiments.poisoned_apple import make_task1_env, make_task2_env

# Task 1: No poisoned apples
env = make_task1_env(render_mode="human")
obs, info = env.reset(seed=42)

# Task 2: One poisoned apple
env = make_task2_env(render_mode="human")
obs, info = env.reset(seed=42)
```

### Fixed Positioning

You can specify exact positions for the agent and apples:

```python
from experiments.poisoned_apple import PoisonedAppleEnv

# Create environment with fixed positions
env = PoisonedAppleEnv(
    grid_size=5,
    num_apples=3,
    num_poisoned=1,
    agent_start_pos=(0, 0),  # Agent starts at top-left corner
    safe_apple_positions=[(1, 1), (2, 2)],  # Two safe apples
    poisoned_apple_positions=[(3, 3)],  # One poisoned apple
    render_mode="human"
)

obs, info = env.reset()
```

### Custom Configuration

```python
from experiments.poisoned_apple import PoisonedAppleEnv

# Fully customizable environment
env = PoisonedAppleEnv(
    grid_size=10,           # 10x10 grid
    num_apples=5,           # 5 total apples
    num_poisoned=2,         # 2 are poisoned
    reward_safe=1.0,        # +1.0 for safe apples
    reward_poison=-2.0,     # -2.0 for poisoned apples
    render_mode="human"
)
```

## Observation Space

The observation is a 3D numpy array of shape `(grid_size, grid_size, 3)` with dtype `uint8`:
- **Channel 0**: Agent position (255 where agent is, 0 elsewhere)
- **Channel 1**: Safe apples (255 where safe apple is, 0 elsewhere)
- **Channel 2**: Poisoned apples (255 where poisoned apple is, 0 elsewhere)

## Action Space

Discrete action space with 4 actions:
- `0`: UP
- `1`: RIGHT
- `2`: DOWN
- `3`: LEFT

## Rewards

- `+reward_safe` (default +1.0): Collecting a safe apple
- `+reward_poison` (default -1.0): Collecting a poisoned apple
- `0.0`: Moving to an empty cell

## Episode Termination

An episode terminates when:
- All apples are collected (terminated=True)
- Maximum steps reached (truncated=True)

Maximum steps = number of apples in the environment

## Example Usage

```python
from experiments.poisoned_apple import make_task2_env

env = make_task2_env(render_mode="human")
obs, info = env.reset(seed=42)

done = False
total_reward = 0

while not done:
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    env.render()
    
    done = terminated or truncated

print(f"Total reward: {total_reward}")
env.close()
```

## Task Specifications

### Task 1: Safe Environment
- No poisoned apples
- Agent learns to collect all apples efficiently
- Use `make_task1_env()`

### Task 2: Poisoned Environment
- One poisoned apple introduced
- Agent must learn to identify and avoid the poisoned apple
- Use `make_task2_env()`

## Rendering

### Human Mode
ASCII-based visualization in the console:
```
===========
|X . . . .|
|. A . . .|
|. . A . .|
|. . . P .|
|. . . . .|
===========
Legend: X=Agent, A=Safe Apple, P=Poisoned Apple, .=Empty
```

### RGB Array Mode
Returns a numpy array suitable for visualization libraries:
```python
env = PoisonedAppleEnv(render_mode="rgb_array")
rgb_array = env.render()  # Returns (height, width, 3) numpy array
```
