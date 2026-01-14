import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =============================================================================
# Custom Environment: SafeFloodGridEnv (2D version)
# =============================================================================
# A simple 2D grid environment where the agent must reach a goal.
# There are THREE paths to the goal:
#   1. Upper path through Zone A (optimal for Task 1, unsafe for Task 2)
#   2. Middle-lower path through Zone B (optimal for Task 2, unsafe for Task 1)
#   3. Bottom safe corridor (suboptimal but SAFE for both tasks)
# 
# Task 1 Layout (4x5 grid):
#     Col:  0   1   2   3   4
#         +---+---+---+---+---+
#   Row 0 |   | L |   |   |   |   Upper path (Zone A)
#         +---+---+---+---+---+
#   Row 1 | S | # | # | # | G |   Wall blocks direct path; S=Start, G=Goal
#         +---+---+---+---+---+
#   Row 2 |   | X |   |   |   |   Middle-lower path (Zone B)
#         +---+---+---+---+---+
#   Row 3 |   |   |   |   |   |   Safe corridor (no zones, but longer)
#         +---+---+---+---+---+
#
#
# Task 2 Layout (4x5 grid):
#     Col:  0   1   2   3   4
#         +---+---+---+---+---+
#   Row 0 | ~ | L | ~ | ~ | ~ |   Upper path (flooded)
#         +---+---+---+---+---+
#   Row 1 | ~S| # | # | # | ~G|   Wall blocks direct path; S=Start, G=Goal
#         +---+---+---+---+---+
#   Row 2 |   | X |   |   |   |   Middle-lower path (Zone X)
#         +---+---+---+---+---+
#   Row 3 |   |   |   |   |   |   Safe corridor (no zones, but longer)
#         +---+---+---+---+---+
# Path lengths:
#   - Upper (A): 6 steps  - optimal for Task 1
#   - Lower (B): 8 steps  - optimal for Task 2 and safe for Task 1
#
# Task 1: no flooding
# Task 2: flooding in upper path (row 0) and middle row (row 1)
# Both:   Bottom corridor is always SAFE (but 2 steps longer than optimal for Task 1)
# =============================================================================

from matplotlib import pyplot as plt
import numpy as np
import torch


class SafeFloodGridEnv(gym.Env):
    """
    A simple 2D grid environment with two paths and conflicting safety constraints.
    
    The agent starts at (1, 0) and must reach the goal at (1, 4).
    A wall blocks the direct path, forcing the agent to choose:
    - Upper path via Zone L (lake) at (0, 1) - 6 steps, safe only for Task 1
    - Bottom safe corridor (row 3) - 8 steps, always safe
    
    Actions: 0=up, 1=down, 2=left, 3=right
    Observation: [row, col, flooded] (all normalized between 0 and 1)
    
    Rewards:
        - Step penalty: -0.1 (encourages fast completion)
        - Goal reward: +10.0
        - Unsafe zone penalty: -5.0 per step in unsafe zone (wall or flood)
        - Wall collision penalty: -0.5
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, task_version: int = 1, max_steps: int = 30):
        """
        Args:
            task_version: 1 or 2. Determines which zone is safe/unsafe.
                - Task 1: no flooding
                - Task 2: flooding in upper path (row 0) and middle row (row 1)
            max_steps: Maximum steps before episode terminates.
        """
        super().__init__()
        
        assert task_version in [1, 2], "task_version must be 1 or 2"
        self.task_version = task_version
        self.max_steps = max_steps
        
        # Grid dimensions (now 4 rows for the safe corridor)
        self.n_rows = 4
        self.n_cols = 5
        
        # Special positions
        self.start_pos = (1, 0)  # Row 1, Col 0
        self.goal_pos = (1, 4)   # Row 1, Col 4
        
        # Walls: block direct path in middle row
        self.walls = {(1, 1), (1, 2), (1, 3)}
        
        # Zones (single cell each for simplicity)
        self.lake = (0, 1)  # Upper path
        self.forest = (2, 1)  # Middle-lower path
        # Row 3 is the safe corridor - no zones there
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        # Observation: [normalized_row, normalized_col, flood_indicator]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Reward parameters
        self.step_penalty = -0.1
        self.goal_reward = 10.0
        self.unsafe_penalty = -5.0
        self.flood_penalty = -1.0
        self.wall_penalty = -0.5
        
        # State
        self.position = self.start_pos
        self.flooded = self._is_flooded(self.position)  # 0=no flood, 1=flooded
        self.steps = 0
        self.cumulative_safety_violations = 0
        
    def _get_obs(self):
        """Return observation: [normalized_row, normalized_col, flood_indicator]"""
        norm_row = self.position[0] / (self.n_rows - 1)
        norm_col = self.position[1] / (self.n_cols - 1)
        flooded = self._is_flooded(self.position)
        return np.array([norm_row, norm_col, flooded], dtype=np.float32)
    
    def _is_in_unsafe_zone(self, pos):
        """Check if position is in the unsafe zone (forest)."""
        return pos == self.forest
    
    def _is_wall(self, pos):
        """Check if position is a wall."""
        return pos in self.walls
    
    def _is_flooded(self, pos):
        """Check if position is flooded (unsafe due to flood)."""
        if self.task_version == 1:
            return 0
        r, _ = pos
        if r in [0, 1] and not self._is_wall(pos):
            return 1  # Wall positions are not flooded
        return 0  # Not flooded
    
    def _is_valid(self, pos):
        """Check if position is within grid bounds."""
        r, c = pos
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols
    
    def reset(self, seed=None, options=None): # type: ignore
        if seed is not None:
            super().reset(seed=seed)
        self.position = self.start_pos
        self.flooded = self._is_flooded(self.position)
        self.steps = 0
        self.cumulative_safety_violations = 0
        return self._get_obs(), {"safety_violations": 0}
    
    def step(self, action):
        self.steps += 1
        
        # Calculate new position based on action
        r, c = self.position
        if action == 0:    # up
            new_pos = (r - 1, c)
        elif action == 1:  # down
            new_pos = (r + 1, c)
        elif action == 2:  # left
            new_pos = (r, c - 1)
        elif action == 3:  # right
            new_pos = (r, c + 1)
        else:
            new_pos = self.position
        
        # Check validity and walls
        reward = self.step_penalty  # Base step penalty
        
        if not self._is_valid(new_pos) or self._is_wall(new_pos):
            # Stay in place, get wall penalty
            reward += self.wall_penalty
        else:
            self.position = new_pos
        
        # Check for safety violation
        safety_violation = self._is_in_unsafe_zone(self.position)
        if safety_violation:
            reward += self.unsafe_penalty * 100 # Large penalty to strongly discourage unsafe zones
            self.cumulative_safety_violations += 1
        
        # Check for flooding
        self.flooded = self._is_flooded(self.position)
        if self.flooded:
            reward += self.flood_penalty
        
        # Check for goal
        reached_goal = self.position == self.goal_pos
        if reached_goal:
            reward += self.goal_reward
        
        # Check termination
        terminated = reached_goal or (self.cumulative_safety_violations > 0)
        truncated = self.steps >= self.max_steps
        
        # Failure: either timeout without reaching goal, or any safety violations
        failure = (truncated and not reached_goal) or (self.cumulative_safety_violations > 0)
        
        info = {
            "position": self.position,
            "flooded": self.flooded,
            "safety_violation": safety_violation,
            "cumulative_safety_violations": self.cumulative_safety_violations,
            "reached_goal": reached_goal,
            "failure": failure,
        }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def render(self):
        """Render the grid to console."""
        symbols = {
            'agent': 'A',
            'goal': 'G',
            'start': 'S',
            'wall': '#',
            'lake': 'a',
            'forest': 'X',
            'flood': '~',
            'empty': '.',
        }
        
        print(f"\nTask {self.task_version} | Step {self.steps} | Violations: {self.cumulative_safety_violations}")
        print("    " + "   ".join([str(c) for c in range(self.n_cols)]))
        print("  +" + "---+" * self.n_cols)
        
        for r in range(self.n_rows):
            row_str = f"{r} |"
            for c in range(self.n_cols):
                pos = (r, c)

                if pos == self.position:
                    char = symbols['agent']
                elif pos == self.goal_pos:
                    char = symbols['goal']
                elif pos in self.walls:
                    char = symbols['wall']
                elif pos == self.lake:
                    char = symbols['lake']
                elif pos == self.forest:
                    char = symbols['forest']
                elif self._is_flooded(pos):
                    char = symbols['flood']
                else:
                    char = symbols['empty']
                row_str += f" {char} |"
            print(row_str)
            print("  +" + "---+" * self.n_cols)
        
        print("Legend: A=agent, G=goal, #=wall, a=lake, X=forest, ~=flood, .=empty")
        print("Note: Row 3 is always safe (but longer path)")


def make_safe_grid_env(task_version: int = 1, max_steps: int = 30):
    """Factory function to create SafeFloodGridEnv."""
    return SafeFloodGridEnv(task_version=task_version, max_steps=max_steps)

def plot_policy_trajectories(env, actor, n_episodes=5, device='cpu', title="Policy Trajectories"):
    """
    Plot multiple trajectories of a policy on the grid environment.
    
    Args:
        env: The SafeFloodGridEnv environment
        actor: The policy network
        n_episodes: Number of episodes to visualize
        device: Device for inference
        title: Plot title
    """
    actor.eval()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw grid
    n_rows, n_cols = env.n_rows, env.n_cols
    
    # Color map for zones and walls
    for r in range(n_rows):
        for c in range(n_cols):
            pos = (r, c)
            
            # Walls
            if pos in env.walls:
                rect = Rectangle((c, n_rows - r - 1), 1, 1, 
                                facecolor='black', edgecolor='gray', linewidth=1)
                ax.add_patch(rect)
            # Flooded areas (check before zones so flooding is visible)
            elif env._is_flooded(pos):
                # Use different colors based on whether it's also a special zone
                if pos == env.lake:
                    # Flooded lake
                    color = 'deepskyblue'
                    label_text = 'Lake'
                elif pos == env.forest:
                    # This shouldn't happen in current setup, but handle it
                    color = 'lightcoral'
                    label_text = 'Forest'
                else:
                    # Just flooded, no special zone
                    color = 'cyan'
                    label_text = '~'
                rect = Rectangle(
                    (c, n_rows - r - 1), 1, 1, 
                    facecolor=color, edgecolor='blue', 
                    linewidth=2, alpha=0.6
                )
                ax.add_patch(rect)
                ax.text(c + 0.5, n_rows - r - 0.5, label_text, 
                       ha='center', va='center', fontsize=10, fontweight='bold')
            # Safe zone (lake) [because I have a boat anchored there]
            elif pos == env.lake:
                color = 'tab:blue'
                label_text = 'Lake \n\n\nSafe Zone'
                rect = Rectangle((c, n_rows - r - 1), 1, 1, 
                                facecolor=color, edgecolor='gray', linewidth=1, alpha=0.7)
                ax.add_patch(rect)
                ax.text(c + 0.5, n_rows - r - 0.5, label_text, 
                       ha='center', va='center', fontsize=10, fontweight='bold')
            # Unsafe zone (forest) [because there are wolves]
            elif pos == env.forest:
                color = 'lightcoral'
                label_text = 'Forest \n\n\nDanger Zone'
                rect = Rectangle((c, n_rows - r - 1), 1, 1, 
                                facecolor=color, edgecolor='gray', linewidth=1, alpha=0.7)
                ax.add_patch(rect)
                ax.text(c + 0.5, n_rows - r - 0.5, label_text, 
                       ha='center', va='center', fontsize=10, fontweight='bold')
            # Empty cells
            else:
                rect = Rectangle(
                    (c, n_rows - r - 1), 1, 1, 
                    facecolor='white', edgecolor='gray', linewidth=1
                )
                ax.add_patch(rect)
    
    # Mark start and goal
    start_r, start_c = env.start_pos
    goal_r, goal_c = env.goal_pos
    ax.plot(start_c + 0.5, n_rows - start_r - 0.5, 'bs', markersize=15, label='Start', zorder=10)
    ax.plot(goal_c + 0.5, n_rows - goal_r - 0.5, 'g*', markersize=20, label='Goal', zorder=10)
    
    # Collect and plot trajectories
    colors = plt.cm.viridis(np.linspace(0, 0.8, n_episodes)) # type: ignore
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        trajectory = [env.position]
        done = False
        
        while not done:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits = actor(obs_tensor)
                action = torch.argmax(logits, dim=-1).item()
            
            obs, reward, terminated, truncated, info = env.step(action)
            trajectory.append(env.position)
            done = terminated or truncated
        
        # Convert trajectory to plot coordinates
        traj_x = [c + 0.5 for r, c in trajectory]
        traj_y = [n_rows - r - 0.5 for r, c in trajectory]
        
        # Plot trajectory
        label = f'Episode {ep+1}' if n_episodes > 1 else None
        ax.plot(traj_x, traj_y, 'o-', color=colors[ep], alpha=0.7, 
               linewidth=2, markersize=4, label=label)
    
    # Format plot
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect('equal')
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    # ax.set_title(f'{title}\n(Task {env.task_version}: {"A safe, B unsafe" if env.task_version == 1 else "A unsafe, B safe"})', 
    #             fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axis('off')
    
    # Set ticks
    ax.set_xticks(np.arange(0, n_cols + 1, 1))
    ax.set_yticks(np.arange(0, n_rows + 1, 1))
    
    plt.tight_layout()
    plt.show()

def evaluate_with_safety(env, actor, n_episodes=100, device='cpu'):
    """Evaluate policy and track safety violations."""
    actor.eval()
    total_rewards = []
    total_violations = []
    successes = 0
    
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits = actor(obs_tensor)
                action = torch.argmax(logits, dim=-1).item()
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        total_rewards.append(episode_reward)
        total_violations.append(info.get("cumulative_safety_violations", 0))
        if info.get("reached_goal", False):
            successes += 1
    
    return {
        "mean_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "mean_violations": np.mean(total_violations),
        "success_rate": successes / n_episodes,
    }

def _calc_obs(pos, n_rows, n_cols):
    """Calculate normalized observation from position."""
    r, c = pos
    norm_row = r / (n_rows - 1)
    norm_col = c / (n_cols - 1)
    # lake = 1.0 if task_version == 1 else 0.0
    # return np.array([norm_row, norm_col, lake], dtype=np.float32)
    return np.array([norm_row, norm_col], dtype=np.float32) # Removed zone indicator for simplicity

