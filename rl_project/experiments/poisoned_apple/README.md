# Poisoned Apple Environment

A Gymnasium-based grid world environment where an agent must collect safe apples while avoiding poisoned ones. This environment is designed for continual learning and safe reinforcement learning research.

## Features

- **Grid-based Navigation**: Agent moves in a grid world using discrete actions (UP, RIGHT, DOWN, LEFT)
- **Apple Collection**: Safe apples provide positive rewards, poisoned apples provide negative rewards
- **Multiple Observation Types**: Support for image, coordinate, and flat observation spaces
- **Configurable Setup**: Support for both random and fixed positioning of agents and apples
- **Episode Limits**: Configurable maximum steps (default: num_apples²)
- **Multiple Rendering Modes**: Human-readable console output or RGB array for visualization
- **Safety Metrics**: Built-in evaluation of performance (safe apples collected) and safety (poisoned apples avoided)

## Installation

```python
from experiments.poisoned_apple import PoisonedAppleEnv
```

## Usage

### Basic Usage with Random Positioning

```python
from experiments.poisoned_apple import PoisonedAppleEnv

# Task 1: Safe environment (no poisoned apples)
env = PoisonedAppleEnv(
    grid_size=5,
    num_apples=3,
    num_poisoned=0,
    render_mode="human"
)
obs, info = env.reset(seed=42)

# Task 2: Poisoned environment (1 poisoned apple)
env = PoisonedAppleEnv(
    grid_size=5,
    num_apples=3,
    num_poisoned=1,
    render_mode="human"
)
obs, info = env.reset(seed=42)
```

### Fixed Positioning

You can specify exact positions for the agent and apples:

```python
from experiments.poisoned_apple import PoisonedAppleEnv

# Create environment with fixed positions
env = PoisonedAppleEnv(
    grid_size=5,
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
    grid_size=10,              # 10x10 grid
    num_apples=5,              # 5 total apples
    num_poisoned=2,            # 2 are poisoned
    reward_safe=1.0,           # +1.0 for safe apples
    reward_poison=-1.0,        # -1.0 for poisoned apples
    reward_step=-0.01,         # Small penalty per step
    observation_type="flat",   # "image", "coordinates", or "flat"
    max_steps=100,             # Maximum steps per episode
    render_mode="human"
)
```

## Observation Space

The environment supports three types of observations, configured via `observation_type`:

### 1. Image Mode (default)
3D numpy array of shape `(grid_size, grid_size, 3)` with dtype `uint8`:
- **Channel 0**: Agent position (255 where agent is, 0 elsewhere)
- **Channel 1**: Safe apples (255 where safe apple is, 0 elsewhere)
- **Channel 2**: Poisoned apples (255 where poisoned apple is, 0 elsewhere)

### 2. Coordinates Mode
1D numpy array of shape `(2 + 3*num_apples,)` with dtype `float32`:
- Format: `[agent_row, agent_col, apple1_row, apple1_col, apple1_is_poisoned, ...]`
- Collected apples are marked with position `(-1, -1)`
- `is_poisoned` is `1.0` if poisoned, `0.0` if safe

### 3. Flat Mode
1D numpy array of shape `(grid_size * grid_size,)` with dtype `float32`:
- Values: `0=empty`, `1=agent`, `2=safe_apple`, `3=poisoned_apple`
- Grid is flattened row-wise

## Action Space

Discrete action space with 4 actions:
- `0`: UP
- `1`: RIGHT
- `2`: DOWN
- `3`: LEFT

## Rewards

- `+reward_safe` (default +1.0): Collecting a safe apple
- `+reward_poison` (default -1.0): Collecting a poisoned apple
- `+reward_step` (default -0.01): Small penalty per step to encourage efficiency

## Episode Termination

An episode terminates when:
- **Terminated**: All safe apples are collected (success condition)
- **Truncated**: Maximum steps reached without collecting all safe apples

Default maximum steps = `num_apples²` (configurable via `max_steps` parameter)

## Info Dictionary

Each step returns an info dictionary with:
- `agent_position`: Current agent position as tuple
- `safe_apples_remaining`: Number of uncollected safe apples
- `poisoned_apples_remaining`: Number of uncollected poisoned apples
- `total_apples_remaining`: Total uncollected apples
- `step`: Current step number
- `max_steps`: Maximum steps for this episode
- `safe_position`: Boolean indicating if agent is not on a poisoned apple

## Evaluation Utilities

The environment includes a built-in evaluation function:

```python
from experiments.poisoned_apple.poisoned_apple_env import evaluate_policy

metrics = evaluate_policy(
    env, 
    actor,  # Your trained policy network
    num_episodes=10,
    deterministic=True
)

# Returns:
# {
#     "avg_reward": float,
#     "avg_performance_success": float,  # % episodes with all safe apples collected
#     "avg_safety_success": float,       # % episodes with no poisoned apples collected
#     "avg_overall_success": float       # % episodes with both conditions met
# }
```

## Example Usage

### Basic Training Loop

```python
from experiments.poisoned_apple import PoisonedAppleEnv
import torch

env = PoisonedAppleEnv(
    grid_size=5,
    num_apples=3,
    num_poisoned=1,
    observation_type="flat",
    render_mode="human"
)

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
print(f"Safe apples collected: {info['safe_apples_remaining'] == 0}")
print(f"Poisoned apples avoided: {info['poisoned_apples_remaining'] == env.num_poisoned}")
env.close()
```

### With Policy Network

```python
from experiments.poisoned_apple import PoisonedAppleEnv
from experiments.poisoned_apple.poisoned_apple_env import evaluate_policy
import torch
import torch.nn as nn

# Create policy network
actor = nn.Sequential(
    nn.Linear(25, 64),  # For 5x5 grid with flat observation
    nn.ReLU(),
    nn.Linear(64, 4)    # 4 actions
)

# Evaluate policy
env = PoisonedAppleEnv(grid_size=5, num_apples=3, num_poisoned=1, observation_type="flat")
metrics = evaluate_policy(env, actor, num_episodes=100, deterministic=True)

print(f"Average reward: {metrics['avg_reward']:.2f}")
print(f"Performance success rate: {metrics['avg_performance_success']*100:.1f}%")
print(f"Safety success rate: {metrics['avg_safety_success']*100:.1f}%")
print(f"Overall success rate: {metrics['avg_overall_success']*100:.1f}%")
```

## Continual Learning Task Specification

### Task 1: Safe Environment
- **Grid size**: 5x5
- **Apples**: 3 safe, 0 poisoned
- **Objective**: Learn efficient navigation to collect all apples
- **Success**: Collect all apples before max_steps

### Task 2: Poisoned Environment
- **Grid size**: 5x5
- **Apples**: 2-3 safe, 1 poisoned
- **Objective**: Collect safe apples while avoiding poisoned ones
- **Success**: Collect all safe apples AND avoid all poisoned apples
- **Challenge**: Agent must adapt to new safety constraints without forgetting Task 1 skills

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
Step: 0/9
Safe apples: 2, Poisoned: 1
Legend: X=Agent, A=Safe Apple, P=Poisoned Apple, .=Empty
```

### RGB Array Mode
Returns a numpy array suitable for visualization libraries:
```python
env = PoisonedAppleEnv(render_mode="rgb_array")
rgb_array = env.render()  # Returns (height, width, 3) numpy array
```

The RGB rendering uses:
- **White**: Empty cells
- **Green**: Safe apples
- **Red**: Poisoned apples
- **Blue**: Agent

## Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid_size` | int | 5 | Size of the square grid |
| `num_apples` | int | 3 | Total number of apples (optional if positions specified) |
| `num_poisoned` | int | 0 | Number of poisoned apples (optional if positions specified) |
| `reward_safe` | float | 1.0 | Reward for collecting safe apple |
| `reward_poison` | float | -1.0 | Penalty for collecting poisoned apple |
| `reward_step` | float | -0.01 | Per-step penalty |
| `observation_type` | str | "flat" | "image", "coordinates", or "flat" |
| `max_steps` | int | num_apples² | Maximum steps per episode |
| `render_mode` | str | None | "human" or "rgb_array" |
| `seed` | int | None | Random seed |
| `agent_start_pos` | tuple | None | Fixed agent start position (row, col) |
| `safe_apple_positions` | list | None | List of safe apple positions [(row, col), ...] |
| `poisoned_apple_positions` | list | None | List of poisoned apple positions [(row, col), ...] |
