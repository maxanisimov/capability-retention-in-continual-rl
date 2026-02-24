"""
Demo script for CustomParkingEnv with exact vehicle placement.

This script demonstrates:
1. Using the registered environment with gymnasium.make()
2. Using direct instantiation
3. Custom spot placement vs random placement
"""

import os
import sys
import numpy as np
import gymnasium
import highway_env
import matplotlib.pyplot as plt

# ── Path setup ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
_RL_DIR = os.path.join(_PROJECT_ROOT, "rl_project")
for p in (_RL_DIR, _PROJECT_ROOT, _SCRIPT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# Import to register the custom environment
from utils.custom_envs import CustomParkingEnv

# ── Output directory ────────────────────────────────────────────────────────
_RESULTS_DIR = os.path.join(_SCRIPT_DIR, "results")


def demo_custom_placement():
    """Demo with custom vehicle spot placement."""
    print("=" * 60)
    print("Demo 1: Custom Vehicle Placement")
    print("=" * 60)

    config = {
        "observation": {
            "type": "KinematicsGoal",
            "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
            "scales": [100, 100, 5, 5, 1, 1],
            "normalize": False,
        },
        "action": {"type": "ContinuousAction"},
        "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
        "success_goal_reward": 0.12,
        "collision_reward": -5,
        "steering_range": np.deg2rad(45),
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "duration": 100,
        "screen_width": 600,
        "screen_height": 300,
        "centering_position": [0.5, 0.5],
        "scaling": 7,
        "controlled_vehicles": 1,
        "vehicles_count": 3,
        # Custom spot indices: ('a', 'b', i) is lower lane; ('b', 'c', i) is upper lane
        "vehicles_spot_indices": [
            ('a', 'b', 0),  # Lower lane, spot 0
            ('a', 'b', 1),  # Lower lane, spot 1
            ('b', 'c', 0),  # Upper lane, spot 0
        ],
        "add_walls": True,
    }

    seed = 100

    # Method 1: Use registered environment
    env = gymnasium.make('custom-parking-v0', render_mode='rgb_array', config=config)
    obs, info = env.reset(seed=seed)

    print(f"Environment created with seed={seed}")
    print(f"Observation shape: {obs['observation'].shape}")
    print(f"Action space: {env.action_space}")
    print(f"Vehicles placed at custom spots: {config['vehicles_spot_indices']}")

    # Render initial state
    frame = env.render()

    plt.figure(figsize=(12, 6))
    plt.imshow(frame)
    plt.title(f"Custom Placement - Initial State (seed={seed})")
    plt.axis('off')
    plt.tight_layout()
    _path = os.path.join(_RESULTS_DIR, "custom_placement_demo.png")
    plt.savefig(_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {_path}")

    env.close()
    return frame


def demo_random_placement():
    """Demo with random vehicle placement (no custom indices)."""
    print("\n" + "=" * 60)
    print("Demo 2: Random Vehicle Placement")
    print("=" * 60)

    config = {
        "observation": {
            "type": "KinematicsGoal",
            "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
            "scales": [100, 100, 5, 5, 1, 1],
            "normalize": False,
        },
        "action": {"type": "ContinuousAction"},
        "controlled_vehicles": 1,
        "vehicles_count": 5,  # More vehicles for interesting layout
        "vehicles_spot_indices": None,  # Random placement
        "add_walls": True,
    }

    seed = 42

    # Method 2: Direct instantiation
    env = CustomParkingEnv(config=config, render_mode='rgb_array')
    obs, info = env.reset(seed=seed)

    print(f"Environment created with seed={seed}")
    print(f"Vehicles placed randomly (vehicles_spot_indices=None)")
    print(f"Number of obstacle vehicles: {config['vehicles_count']}")

    # Render initial state
    frame = env.render()

    plt.figure(figsize=(12, 6))
    plt.imshow(frame)
    plt.title(f"Random Placement - Initial State (seed={seed})")
    plt.axis('off')
    plt.tight_layout()
    _path = os.path.join(_RESULTS_DIR, "random_placement_demo.png")
    plt.savefig(_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {_path}")

    env.close()
    return frame


def demo_comparison():
    """Compare same seed with standard parking-v0 vs custom-parking-v0."""
    print("\n" + "=" * 60)
    print("Demo 3: Comparison - Standard vs Custom Environment")
    print("=" * 60)

    config = {
        "controlled_vehicles": 1,
        "vehicles_count": 4,
        "add_walls": True,
    }

    seed = 123

    # Standard environment
    env_standard = gymnasium.make('parking-v0', render_mode='rgb_array', config=config)
    obs_std, _ = env_standard.reset(seed=seed)
    frame_std = env_standard.render()
    env_standard.close()

    # Custom environment with specific placement
    config_custom = config.copy()
    config_custom["vehicles_spot_indices"] = [
        ('a', 'b', 2),
        ('a', 'b', 3),
        ('b', 'c', 2),
        ('b', 'c', 3),
    ]

    env_custom = gymnasium.make('custom-parking-v0', render_mode='rgb_array', config=config_custom)
    obs_custom, _ = env_custom.reset(seed=seed)
    frame_custom = env_custom.render()
    env_custom.close()

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].imshow(frame_std)
    axes[0].set_title(f"Standard ParkingEnv (random placement)\nseed={seed}")
    axes[0].axis('off')

    axes[1].imshow(frame_custom)
    axes[1].set_title(f"CustomParkingEnv (exact placement)\nseed={seed}")
    axes[1].axis('off')

    plt.tight_layout()
    _path = os.path.join(_RESULTS_DIR, "comparison_demo.png")
    plt.savefig(_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {_path}")
    print("\nNotice how custom placement is deterministic regardless of seed!")


if __name__ == "__main__":
    os.makedirs(_RESULTS_DIR, exist_ok=True)

    # Run demos
    demo_custom_placement()
    demo_random_placement()
    demo_comparison()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)

    plt.show()
