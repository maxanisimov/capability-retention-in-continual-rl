"""
Poisoned Apple Environment - A Gymnasium-based grid world where an agent
must collect safe apples while avoiding poisoned ones.

The environment features:
- Grid-based navigation
- Multiple apples (some may be poisoned)
- Limited steps (equal to number of apples)
- Rewards for collecting safe apples, penalties for poisoned ones
"""

from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import gymnasium
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import torch

class PoisonedAppleEnv(gymnasium.Env):
    """
    A grid world environment where an agent collects apples.
    
    The agent navigates a grid to collect apples. Some apples may be poisoned.
    The episode has a maximum number of steps equal to the number of apples.
    
    Observation Space:
        Image mode (default):
            Box(0, 255, (grid_size, grid_size, 3), dtype=uint8)
            - Channel 0: Agent position (255 where agent is, 0 elsewhere)
            - Channel 1: Safe apples (255 where safe apple is, 0 elsewhere)
            - Channel 2: Poisoned apples (255 where poisoned apple is, 0 elsewhere)
        
        Coordinates mode:
            Box(-1, grid_size-1, (2 + 3*num_apples,), dtype=float32)
            - [agent_row, agent_col, apple1_row, apple1_col, apple1_is_poisoned, ..., appleN_row, appleN_col, appleN_is_poisoned]
            - Collected apples are marked with position (-1, -1)
            - is_poisoned is 1.0 if poisoned, 0.0 if safe
    
    Action Space:
        Discrete(4): [UP, RIGHT, DOWN, LEFT]
    
    Rewards:
        - +1.0 for collecting a safe apple
        - -1.0 for collecting a poisoned apple
        - 0.0 for moving to an empty cell
    
    Episode Termination:
        - All apples are collected
        - Maximum steps reached (equal to number of apples)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    # Actions
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    
    def __init__(
        self,
        grid_size: int = 5,
        num_apples: Optional[int] = 3,
        num_poisoned: Optional[int] = 0,
        reward_safe: float = 1.0,
        reward_poison: float = -1.0,
        reward_step: float = -0.01,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        agent_start_pos: Optional[Tuple[int, int]] = None,
        safe_apple_positions: Optional[list] = None,
        poisoned_apple_positions: Optional[list] = None,
        observation_type: str = "flat",
        max_steps: Optional[int] = None
    ):
        """
        Initialize the Poisoned Apple Environment.
        
        Args:
            grid_size: Size of the square grid (default: 5x5)
            num_apples: Total number of apples in the environment
            num_poisoned: Number of poisoned apples (must be <= num_apples)
            reward_safe: Reward for collecting a safe apple
            reward_poison: Reward (penalty) for collecting a poisoned apple
            render_mode: Mode for rendering ("human" or "rgb_array")
            seed: Random seed for reproducibility
            agent_start_pos: Fixed starting position for agent as (row, col). If None, random.
            safe_apple_positions: List of fixed positions for safe apples as [(row, col), ...]. If None, random.
            poisoned_apple_positions: List of fixed positions for poisoned apples as [(row, col), ...]. If None, random.
            observation_type: Type of observation - "image" (3D grid), "coordinates" (flat array of positions), or "flat" (flattened image)
            max_steps: Maximum number of steps per episode. If None, defaults to num_apples squared.
        """
        super().__init__()

        if (poisoned_apple_positions is None):
            assert num_poisoned is not None, "num_poisoned must be specified if positions are not fixed"
        else:
            # NOTE: the list takes precedence over num_poisoned (if both provided)
            num_poisoned = len(poisoned_apple_positions)

        if (safe_apple_positions is None):
            assert num_apples is not None, "num_apples must be specified if positions are not fixed"
        else:
            # NOTE: the list takes precedence over num_apples (if both provided)
            if (poisoned_apple_positions is None):
                num_apples = len(safe_apple_positions) + num_poisoned
            else:
                num_apples = len(safe_apple_positions) + len(poisoned_apple_positions)
        
        assert num_poisoned <= num_apples, "Number of poisoned apples cannot exceed total apples"
        assert grid_size >= 3, "Grid size must be at least 3x3"
        assert observation_type in ["image", "coordinates", "flat"], "observation_type must be 'image', 'coordinates', or 'flat'"

        # Validate fixed positions if provided
        if safe_apple_positions is not None and poisoned_apple_positions is not None:
            assert len(safe_apple_positions) + len(poisoned_apple_positions) == num_apples, \
                "Total fixed positions must equal num_apples"
            assert len(poisoned_apple_positions) == num_poisoned, \
                "Number of poisoned positions must equal num_poisoned"
        
        self.grid_size = grid_size
        self.num_apples = num_apples
        self.num_poisoned = num_poisoned
        self.num_safe_apples = num_apples - num_poisoned
        self.reward_safe = reward_safe
        self.reward_poison = reward_poison
        self.reward_step = reward_step
        self.render_mode = render_mode
        self.observation_type = observation_type
        
        # Store fixed positions (if provided)
        self.agent_start_pos = agent_start_pos
        self.safe_apple_positions = safe_apple_positions
        self.poisoned_apple_positions = poisoned_apple_positions
        
        # Define observation space based on observation type
        if observation_type == "image":
            # Channel 0: Agent position, Channel 1: Safe apples, Channel 2: Poisoned apples
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(grid_size, grid_size, 3), dtype=np.uint8
            )
        elif observation_type == "coordinates":
            # Format: [agent_row, agent_col, apple1_row, apple1_col, apple1_is_poisoned, ..., appleN_row, appleN_col, appleN_is_poisoned]
            # Collected apples marked as -1, is_poisoned is 1.0 or 0.0
            self.observation_space = spaces.Box(
                low=-1, high=grid_size-1, shape=(2 + 3*num_apples,), dtype=np.float32
            )
        else:  # flat
            # Flattened grid: 0=empty, 1=agent, 2=safe_apple, 3=poisoned_apple
            self.observation_space = spaces.Box(
                low=0, high=3, shape=(grid_size * grid_size,), dtype=np.float32
            )
        
        # Define action space: 4 directions (UP, RIGHT, DOWN, LEFT)
        self.action_space = spaces.Discrete(4)
        
        # Initialize random number generator
        self._np_random = None
        if seed is not None:
            self.seed(seed)
        
        # Episode state variables
        self.agent_pos = None
        self.safe_apples = None
        self.poisoned_apples = None
        self.max_steps = max_steps if max_steps is not None else self.num_apples**2 # TODO: self.grid_size*self.num_safe_apples can be a good choice; but smaller max_steps makes training easier
        self.current_step = 0
        
        # For coordinate observations, maintain ordered list of initial apple positions
        self.initial_apple_positions = None  # Will store list of (pos, is_poisoned) tuples
        
    def seed(self, seed: Optional[int] = None):
        """Set the seed for random number generation."""
        self._np_random = np.random.RandomState(seed) # type: ignore
        return [seed]
    
    def reset( # type: ignore
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for this episode
            options: Additional options (not used)
        
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self.seed(seed)
        
        if self._np_random is None:
            self._np_random = np.random.RandomState() # type: ignore
        
        # Reset step counter
        self.current_step = 0
        
        # Use fixed positions if provided, otherwise generate random positions
        if self.agent_start_pos is not None:
            # Use fixed agent position
            self.agent_pos = np.array(self.agent_start_pos)
        else:
            # Generate random agent position
            available_positions = [
                (i, j) for i in range(self.grid_size) for j in range(self.grid_size)
            ]
            agent_idx = self._np_random.choice(len(available_positions)) # type: ignore
            self.agent_pos = np.array(available_positions[agent_idx])
        
        # Set apple positions
        if self.safe_apple_positions is not None and self.poisoned_apple_positions is not None:
            # Use fixed apple positions
            self.safe_apples = set(self.safe_apple_positions)
            self.poisoned_apples = set(self.poisoned_apple_positions)
        else:
            # Generate random apple positions
            available_positions = [
                (i, j) for i in range(self.grid_size) for j in range(self.grid_size)
                if not (i == self.agent_pos[0] and j == self.agent_pos[1])
            ]
            
            # Randomly select positions for apples
            selected_positions = self._np_random.choice(  # type: ignore
                len(available_positions), 
                size=self.num_apples,
                replace=False
            )
            
            apple_positions = [available_positions[i] for i in selected_positions]
            
            # Randomly assign which apples are poisoned
            poisoned_indices = self._np_random.choice( # type: ignore
                self.num_apples,
                size=self.num_poisoned,
                replace=False
            )
            
            self.safe_apples = set()
            self.poisoned_apples = set()
            
            for i, pos in enumerate(apple_positions):
                if i in poisoned_indices:
                    self.poisoned_apples.add(pos)
                else:
                    self.safe_apples.add(pos)
        
        # Store initial apple positions in order (for coordinate observations)
        # Format: list of (position, is_poisoned) tuples
        self.initial_apple_positions = []
        for pos in sorted(self.safe_apples):
            self.initial_apple_positions.append((pos, False))
        for pos in sorted(self.poisoned_apples):
            self.initial_apple_positions.append((pos, True))
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT)
        
        Returns:
            observation: New observation
            reward: Reward received
            terminated: Whether episode is terminated
            truncated: Whether episode is truncated (max steps)
            info: Additional information
        """
        # Update step counter
        self.current_step += 1
        
        # Calculate new position based on action
        assert self.agent_pos is not None, "Agent position not initialized"
        new_pos = self.agent_pos.copy()
        
        if action == self.UP:
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == self.RIGHT:
            new_pos[1] = min(self.grid_size - 1, new_pos[1] + 1)
        elif action == self.DOWN:
            new_pos[0] = min(self.grid_size - 1, new_pos[0] + 1)
        elif action == self.LEFT:
            new_pos[1] = max(0, new_pos[1] - 1)
        
        # Update agent position
        self.agent_pos = new_pos
        
        # Check if agent collected an apple
        reward = self.reward_step  # Small step penalty to encourage efficiency
        agent_tuple = tuple(self.agent_pos)
        
        assert self.safe_apples is not None and self.poisoned_apples is not None, "Apples not initialized"
        if agent_tuple in self.safe_apples:
            self.safe_apples.remove(agent_tuple)
            reward += self.reward_safe
        elif agent_tuple in self.poisoned_apples:
            self.poisoned_apples.remove(agent_tuple)
            reward += self.reward_poison
        
        # Check termination conditions
        # all_apples_collected = len(self.safe_apples) == 0 and len(self.poisoned_apples) == 0
        all_safe_apples_collected = len(self.safe_apples) == 0
        max_steps_reached = False
        if self.max_steps is not None:
            max_steps_reached = self.current_step >= self.max_steps
        
        terminated = all_safe_apples_collected
        truncated = max_steps_reached and not terminated
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Generate the current observation.
        
        Returns:
            observation: numpy array (format depends on observation_type)
        """
        if self.observation_type == "image":
            return self._get_image_observation()
        elif self.observation_type == "coordinates":
            return self._get_coordinate_observation()
        else:  # flat
            return self._get_flat_observation()
    
    def _get_image_observation(self) -> np.ndarray:
        """
        Generate image-based observation.
        
        Returns:
            observation: 3D numpy array (grid_size, grid_size, 3)
        """
        obs = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        
        # Channel 0: Agent position
        assert self.agent_pos is not None, "Agent position not initialized"
        obs[self.agent_pos[0], self.agent_pos[1], 0] = 255
        
        # Channel 1: Safe apples
        assert self.safe_apples is not None, "Safe apples not initialized"
        for pos in self.safe_apples:
            obs[pos[0], pos[1], 1] = 255
        
        # Channel 2: Poisoned apples
        assert self.poisoned_apples is not None, "Poisoned apples not initialized"
        for pos in self.poisoned_apples:
            obs[pos[0], pos[1], 2] = 255
        
        return obs
    
    def _get_coordinate_observation(self) -> np.ndarray:
        """
        Generate coordinate-based observation.
        
        Returns:
            observation: 1D numpy array [agent_row, agent_col, apple1_row, apple1_col, apple1_is_poisoned, ...]
                        Collected apples are marked as (-1, -1)
                        is_poisoned is 1.0 if poisoned, 0.0 if safe
        """
        obs = np.zeros(2 + 3*self.num_apples, dtype=np.float32)
        
        # Agent position
        assert self.agent_pos is not None, "Agent position not initialized"
        obs[0] = self.agent_pos[0]
        obs[1] = self.agent_pos[1]
        
        # Apple positions (in initial order)
        assert self.initial_apple_positions is not None, "Initial apple positions not set"
        for i, (pos, is_poisoned) in enumerate(self.initial_apple_positions):
            idx = 2 + 3*i
            # Check if apple is still present
            if is_poisoned:
                if pos in self.poisoned_apples:
                    obs[idx] = pos[0]
                    obs[idx + 1] = pos[1]
                else:
                    obs[idx] = -1
                    obs[idx + 1] = -1
                obs[idx + 2] = 1.0  # Poisoned
            else:
                if pos in self.safe_apples:
                    obs[idx] = pos[0]
                    obs[idx + 1] = pos[1]
                else:
                    obs[idx] = -1
                    obs[idx + 1] = -1
                obs[idx + 2] = 0.0  # Safe
        
        return obs
    
    def _get_flat_observation(self) -> np.ndarray:
        """
        Generate flattened grid observation.
        
        Returns:
            observation: 1D numpy array (grid_size * grid_size,)
                        Values: 0=empty, 1=agent, 2=safe_apple, 3=poisoned_apple
        """
        obs = np.zeros(self.grid_size * self.grid_size, dtype=np.float32)

        # Place safe apples (value 2)
        assert self.safe_apples is not None, "Safe apples not initialized"
        for pos in self.safe_apples:
            idx = pos[0] * self.grid_size + pos[1]
            obs[idx] = 2.0
        
        # Place poisoned apples (value 3)
        assert self.poisoned_apples is not None, "Poisoned apples not initialized"
        for pos in self.poisoned_apples:
            idx = pos[0] * self.grid_size + pos[1]
            obs[idx] = 3.0
        
        # Place agent (value 1) - overwrites apple if on same position
        assert self.agent_pos is not None, "Agent position not initialized"
        agent_idx = self.agent_pos[0] * self.grid_size + self.agent_pos[1]
        obs[agent_idx] = 1.0
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Generate additional information about the current state.
        
        Returns:
            info: Dictionary with state information
        """
        assert self.agent_pos is not None, "Agent position not initialized"
        assert self.safe_apples is not None, "Safe apples not initialized"
        assert self.poisoned_apples is not None, "Poisoned apples not initialized"
        return {
            "agent_position": tuple(self.agent_pos),
            "safe_apples_remaining": len(self.safe_apples),
            "poisoned_apples_remaining": len(self.poisoned_apples),
            "total_apples_remaining": len(self.safe_apples) + len(self.poisoned_apples),
            "step": self.current_step,
            "max_steps": self.max_steps,
            "safe": tuple(self.agent_pos) not in self.poisoned_apples # whether agent is currently on a poisoned apple
        }
    
    def render(self):
        """
        Render the environment.
        
        Returns:
            If render_mode is "rgb_array", returns the RGB array.
            If render_mode is "human", prints the environment to console.
        """
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
    
    def _render_human(self):
        """Render the environment in human-readable format to console."""
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Place apples
        assert self.safe_apples is not None, "Safe apples not initialized"
        for pos in self.safe_apples:
            grid[pos[0]][pos[1]] = 'A'
        
        assert self.poisoned_apples is not None, "Poisoned apples not initialized"
        for pos in self.poisoned_apples:
            grid[pos[0]][pos[1]] = 'P'
        
        # Place agent (overrides apples if on same position)
        assert self.agent_pos is not None, "Agent position not initialized"
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'X'
        
        # Print grid
        print("\n" + "="*(self.grid_size * 2 + 1))
        for row in grid:
            print("|" + " ".join(row) + "|")
        print("="*(self.grid_size * 2 + 1))
        if self.max_steps is not None:
            print(f"Step: {self.current_step}/{self.max_steps}")
        else:
            print(f"Step: {self.current_step}/∞")
        print(f"Safe apples: {len(self.safe_apples)}, Poisoned: {len(self.poisoned_apples)}")
        print(f"Legend: X=Agent, A=Safe Apple, P=Poisoned Apple, .=Empty\n")
    
    def _render_rgb_array(self) -> np.ndarray:
        """
        Render the environment as an RGB array.
        
        Returns:
            RGB array of shape (grid_size*cell_size, grid_size*cell_size, 3)
        """
        cell_size = 50
        img_size = self.grid_size * cell_size
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255  # White background
        
        # Draw grid lines
        for i in range(self.grid_size + 1):
            # Horizontal lines
            img[i * cell_size, :] = 200
            # Vertical lines
            img[:, i * cell_size] = 200
        
        # Draw safe apples (green)
        assert self.safe_apples is not None, "Safe apples not initialized"
        for pos in self.safe_apples:
            y, x = pos[0] * cell_size, pos[1] * cell_size
            img[y+10:y+cell_size-10, x+10:x+cell_size-10] = [0, 255, 0]
        
        # Draw poisoned apples (red)
        assert self.poisoned_apples is not None, "Poisoned apples not initialized"
        for pos in self.poisoned_apples:
            y, x = pos[0] * cell_size, pos[1] * cell_size
            img[y+10:y+cell_size-10, x+10:x+cell_size-10] = [255, 0, 0]
        
        # Draw agent (blue)
        assert self.agent_pos is not None, "Agent position not initialized"
        y, x = self.agent_pos[0] * cell_size, self.agent_pos[1] * cell_size
        img[y+15:y+cell_size-15, x+15:x+cell_size-15] = [0, 0, 255]
        
        return img
    
    def close(self):
        """Clean up resources."""
        pass

################################################################
#### Utlity functions ##########################################
################################################################
def evaluate_policy(env, actor, num_episodes=10, max_steps=None, deterministic=True):
    """
    Evaluate the policy over multiple episodes and return average reward.
    
    Args:
        env: The environment
        actor: Trained actor network
        num_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode (default: env.max_steps)
        deterministic: Whether to use deterministic actions (default: True)

    Returns:
        metrics_dct: Dictionary with average reward and success metrics
    """
    total_rewards = []
    episode_performance = [] # proportion of safe apples collected
    episode_safety = [] # proportion of poisoned apples avoided
    episode_successes = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        while not done:
            # Get action from actor
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action_logits = actor(obs_tensor)
                if deterministic:
                    action = torch.argmax(action_logits, dim=-1).item()
                else:
                    action_probs = torch.softmax(action_logits, dim=-1)
                    action = torch.multinomial(action_probs, num_samples=1).item()
            
            # Take step
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        
        total_rewards.append(episode_reward)
        cur_all_safe_apples_collected = len(env.safe_apples) == 0 # performance metric
        cur_no_poisoned_apples_collected = len(env.poisoned_apples) == env.num_poisoned # safety metric

        episode_success = cur_all_safe_apples_collected and cur_no_poisoned_apples_collected
        # episode_performance.append(cur_all_safe_apples_collected)
        safe_apples_collected_prop = (env.num_safe_apples - len(env.safe_apples)) / env.num_safe_apples
        episode_performance.append(safe_apples_collected_prop)
        poisoned_apples_avoided_prop = (len(env.poisoned_apples)) / env.num_poisoned if env.num_poisoned > 0 else 1.0
        episode_safety.append(poisoned_apples_avoided_prop)
        episode_successes.append(episode_success)
        # print(f"Episode {episode + 1}: Total Reward: {episode_reward:.2f}, "
        #       f"All Safe Apples Collected: {cur_all_safe_apples_collected}, "
        #       f"No Poisoned Apples Collected: {cur_no_poisoned_apples_collected}, "
        #       f"Success: {episode_success}")
    avg_reward = sum(total_rewards) / num_episodes
    avg_performance_success = sum(episode_performance) / num_episodes
    avg_safety_success = sum(episode_safety) / num_episodes
    avg_overall_success = sum(episode_successes) / num_episodes

    metrics_dct = {
        "avg_reward": avg_reward,
        "avg_performance_success": avg_performance_success,
        "avg_safety_success": avg_safety_success,
        "avg_overall_success": avg_overall_success
    }
    return metrics_dct


################################ 
#### HELPER FUNCTIONS ##########
################################

##### VISUALISATION UTILS ######
def visualize_agent_trajectory(
        env, actor, num_episodes=1, max_steps=None, 
        env_name=None, cfg_name=None, actor_name=None, save_dir=None
    ):
    """
    Visualize the trained agent's trajectory in the environment.
    
    Args:
        env: The environment
        actor: Trained actor network
        num_episodes: Number of episodes to visualize
        max_steps: Maximum steps per episode (default: env.max_steps)
        env_name: Optional name for the environment (used in plot titles and filenames)
        cfg_name: Optional configuration name (used in filenames)
        actor_name: Optional actor name (used in filenames)
        save_dir: Optional directory to save plots. If None, plots are only displayed.
    """
    # if max_steps is None:
    #     max_steps = env.max_steps
    # if max_steps is None:
    #     max_steps = np.inf
    max_steps = np.inf
    
    action_names = ["UP", "RIGHT", "DOWN", "LEFT"]
    
    for episode in range(num_episodes):
        print(f"\n{'='*50}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print('='*50)
        
        obs, info = env.reset()
        trajectory = []
        rewards_list = []
        actions_list = []
        
        # Store initial state
        trajectory.append({
            'agent_pos': tuple(env.agent_pos),
            'safe_apples': set(env.safe_apples),
            'poisoned_apples': set(env.poisoned_apples)
        })
        
        done = False
        step = 0
        total_reward = 0
        
        while not done:
            # Get action from actor
            import torch
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action_logits = actor(obs_tensor)
                action = torch.argmax(action_logits, dim=1).item()
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action) # type: ignore
            done = terminated or truncated
            total_reward += reward
            
            # Store trajectory
            trajectory.append({
                'agent_pos': tuple(env.agent_pos),
                'safe_apples': set(env.safe_apples),
                'poisoned_apples': set(env.poisoned_apples)
            })
            rewards_list.append(reward)
            actions_list.append(action)
            
            action_name = action_names[action] # type: ignore
            print(f"Step {step + 1}: {action_name}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
            step += 1

            if step >= max_steps:
                print("Reached maximum steps for this episode.")
                break
        
        print(f"\nEpisode finished! Total reward: {total_reward:.2f}")
        print(f"Apples remaining: {info['safe_apples_remaining']} safe, {info['poisoned_apples_remaining']} poisoned")
        
        # Plot trajectory
        plot_trajectory(
            env, trajectory, rewards_list, actions_list, 
            episode_num=episode + 1 if num_episodes > 1 else None, 
            env_name=env_name, cfg_name=cfg_name, actor_name=actor_name, save_dir=save_dir
        )
    
    if save_dir is None:
        plt.show()

def plot_trajectory(
        env, trajectory, rewards_list, actions_list, 
        episode_num=None, env_name=None, cfg_name=None, actor_name=None, save_dir=None
    ):
    """
    Plot a single trajectory as a static image.
    
    Args:
        env: The environment
        trajectory: List of states
        rewards_list: List of rewards
        actions_list: List of actions
        episode_num: Episode number for title
        env_name: Optional environment name for title
        cfg_name: Optional configuration name for filename
        actor_name: Optional actor name for filename
        save_dir: Optional directory to save the plot. If None, plot is only displayed.
    """
    grid_size = env.grid_size
    num_steps = len(trajectory)
    
    # Create figure with subplots for each step
    cols = min(5, num_steps)
    rows = (num_steps + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    if num_steps == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    action_names = ["↑", "→", "↓", "←"]
    
    for step_idx, state in enumerate(trajectory):
        row = step_idx // cols
        col = step_idx % cols
        ax = axes[row, col]
        
        # Create grid
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_aspect('equal')
        ax.grid(True, linewidth=0.5, alpha=0.3)
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.tick_params(labelsize=12)
        ax.invert_yaxis()
        
        # Draw safe apples (green circles)
        for pos in state['safe_apples']:
            circle = patches.Circle((pos[1], pos[0]), 0.3, color='green', alpha=0.6)
            ax.add_patch(circle)
            ax.text(pos[1], pos[0], 'A', ha='center', va='center', 
                   fontsize=14, fontweight='bold', color='white')
        
        # Draw poisoned apples (red circles)
        for pos in state['poisoned_apples']:
            circle = patches.Circle((pos[1], pos[0]), 0.3, color='red', alpha=0.6)
            ax.add_patch(circle)
            ax.text(pos[1], pos[0], 'P', ha='center', va='center',
                   fontsize=14, fontweight='bold', color='white')
        
        # Draw agent (blue square)
        agent_pos = state['agent_pos']
        rect = patches.Rectangle((agent_pos[1] - 0.35, agent_pos[0] - 0.35), 
                                0.7, 0.7, color='blue', alpha=0.8)
        ax.add_patch(rect)
        ax.text(agent_pos[1], agent_pos[0], '●', ha='center', va='center',
               fontsize=20, fontweight='bold', color='white')
        
        # Title for each step
        if step_idx == 0:
            ax.set_title(f'Start', fontsize=13, fontweight='bold')
        else:
            action = action_names[actions_list[step_idx - 1]]
            reward = rewards_list[step_idx - 1]
            reward_color = 'green' if reward > 0 else ('red' if reward < 0 else 'gray')
            ax.set_title(f'Step {step_idx}: {action} (r={reward:.2f})', 
                        fontsize=13, fontweight='bold', color=reward_color)
            
        # Turn off axis ticks and labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(left=False, bottom=False)
    
    # Hide empty subplots
    for step_idx in range(num_steps, rows * cols):
        row = step_idx // cols
        col = step_idx % cols
        axes[row, col].axis('off')
    
    suptitle = ''
    if cfg_name is not None:
        suptitle = suptitle + cfg_name
    if env_name is not None:
        suptitle = suptitle + ' - ' + env_name
    if actor_name is not None:
        suptitle = suptitle + ' - ' + actor_name
    if episode_num is not None:
        suptitle = suptitle + ' - ' + f'Episode {episode_num}'
    fig.suptitle(suptitle, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the figure if save_dir is provided
    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
    
    if save_dir is not None:
        filename_parts = []
        assert not (cfg_name is None and actor_name is None and env_name is None), "At least one of cfg_name, actor_name, or env_name must be provided for filename."
        if cfg_name is not None:
            filename_parts.append(cfg_name)
        if env_name is not None:
            clean_env_name = env_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            filename_parts.append(clean_env_name)
        if actor_name is not None:
            clean_actor_name = actor_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            filename_parts.append(clean_actor_name)
        if episode_num is not None:
            filename_parts.append(f"episode_{episode_num}")

        filename = "_".join(filename_parts) + ".png"
        filepath = os.path.join(save_dir, filename) # type: ignore
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {filepath}")
        plt.close(fig)

###### SAFETY UTILS ######
def get_observation(
    agent_position: Tuple[int, int],
    safe_apple_positions: set,
    poisoned_apple_positions: set,
    grid_size: int,
    observation_type: str = "flat"
) -> np.ndarray:
    """
    Generate an observation for a given state without modifying the environment.
    
    Args:
        agent_position: Tuple of (row, col) for agent position
        safe_apple_positions: Set of tuples (row, col) for safe apples
        poisoned_apple_positions: Set of tuples (row, col) for poisoned apples
        grid_size: Size of the grid
        observation_type: Type of observation ("flat", "image", or "coordinates")
    
    Returns:
        Observation as numpy array
    
    Raises:
        NotImplementedError: If observation_type is not "flat"
    """
    if observation_type == "flat":
        obs = np.zeros(grid_size * grid_size, dtype=np.float32)
        
        # Place safe apples (value 2)
        for pos in safe_apple_positions:
            idx = pos[0] * grid_size + pos[1]
            obs[idx] = 2.0
        
        # Place poisoned apples (value 3)
        for pos in poisoned_apple_positions:
            idx = pos[0] * grid_size + pos[1]
            obs[idx] = 3.0
        
        # Place agent (value 1) - overwrites apple if on same position
        agent_idx = agent_position[0] * grid_size + agent_position[1]
        obs[agent_idx] = 1.0
        
        return obs
    elif observation_type == "image":
        raise NotImplementedError("Image observation type not yet implemented for get_observation")
    elif observation_type == "coordinates":
        raise NotImplementedError("Coordinates observation type not yet implemented for get_observation")
    else:
        raise ValueError(f"Unknown observation type: {observation_type}")


def get_all_unsafe_state_action_pairs(env: PoisonedAppleEnv) -> list:
    """
    Generate all unsafe state-action pairs for the given environment configuration.
    
    An unsafe state-action pair (s, a) is one where taking action a in state s
    results in the agent moving to a location containing a poisoned apple.
    
    Args:
        env: A PoisonedAppleEnv instance (should be initialized/reset to establish apple positions)
    
    Returns:
        List of tuples: [(state, action), ...] where:
            - state is a numpy array in the format specified by env.observation_type
            - action is an integer (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
    
    Note:
        Currently only supports "flat" observation type. Raises NotImplementedError for other types.
    
    Example:
        env = PoisonedAppleEnv(grid_size=5, num_apples=3, num_poisoned=1, observation_type="flat")
        env.reset()
        unsafe_pairs = get_all_unsafe_state_action_pairs(env)
        # Returns [(state_array, action), ...] for all unsafe transitions
    """
    if env.poisoned_apples is None or env.safe_apples is None:
        raise ValueError("Environment must be reset before generating unsafe pairs")
    
    unsafe_state_action_pairs = []
    poisoned_positions = set(env.poisoned_apples)
    safe_positions = set(env.safe_apples)
    
    # For each possible agent position on the grid
    for row in range(env.grid_size):
        for col in range(env.grid_size):
            # For each possible action
            for action in range(4):  # 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
                # Calculate where the agent would move
                new_row, new_col = row, col
                
                if action == PoisonedAppleEnv.UP:
                    new_row = max(0, row - 1)
                elif action == PoisonedAppleEnv.RIGHT:
                    new_col = min(env.grid_size - 1, col + 1)
                elif action == PoisonedAppleEnv.DOWN:
                    new_row = min(env.grid_size - 1, row + 1)
                elif action == PoisonedAppleEnv.LEFT:
                    new_col = max(0, col - 1)
                
                # Check if new position is a poisoned apple
                if (new_row, new_col) in poisoned_positions:
                    # Generate observation for this state using helper function
                    state = get_observation(
                        agent_position=(row, col),
                        safe_apple_positions=safe_positions,
                        poisoned_apple_positions=poisoned_positions,
                        grid_size=env.grid_size,
                        observation_type=env.observation_type
                    )
                    unsafe_state_action_pairs.append((state, action))
    
    return unsafe_state_action_pairs

def generate_all_observations_env(env, observation_type='flat'):
    """
    Generate all possible observations for env1.
    Since apple positions are fixed, we only need to iterate over agent positions.
    
    Args:
        env: The environment
        observation_type: Type of observation ('flat', 'image', or 'coordinates')
    
    Returns:
        all_observations: numpy array of shape (grid_size^2, obs_dim)
        agent_positions: list of (row, col) tuples for each observation
    """
    
    env.poisoned_apple_positions
    
    all_observations = []
    agent_positions = []
    
    # Iterate over all possible agent positions
    for row in range(env.grid_size):
        for col in range(env.grid_size):
            agent_pos = (row, col)
            
            # Generate observation based on type
            if observation_type == 'flat':
                obs = np.zeros(env.grid_size * env.grid_size, dtype=np.float32)
                
                # Place safe apples (value 2)
                for pos in env.safe_apple_positions:
                    idx = pos[0] * env.grid_size + pos[1]
                    obs[idx] = 2.0
                
                # Place poisoned apples (value 3)
                for pos in env.poisoned_apple_positions:
                    idx = pos[0] * env.grid_size + pos[1]
                    obs[idx] = 3.0
                
                # Place agent (value 1)
                agent_idx = agent_pos[0] * env.grid_size + agent_pos[1]
                obs[agent_idx] = 1.0
                
            elif observation_type == 'image':
                obs = np.zeros((env.grid_size, env.grid_size, 3), dtype=np.uint8)
                
                # Channel 0: Agent position
                obs[agent_pos[0], agent_pos[1], 0] = 255
                
                # Channel 1: Safe apples
                for pos in env.safe_apple_positions:
                    obs[pos[0], pos[1], 1] = 255
                
                # Channel 2: Poisoned apples
                for pos in env.poisoned_apple_positions:
                    obs[pos[0], pos[1], 2] = 255
                
                obs = obs.flatten()  # Flatten for storage
                
            elif observation_type == 'coordinates':
                num_apples = len(env.safe_apple_positions) + len(env.poisoned_apple_positions)
                obs = np.zeros(2 + 3*num_apples, dtype=np.float32)
                
                # Agent position
                obs[0] = agent_pos[0]
                obs[1] = agent_pos[1]
                
                # Combine and sort apples (safe first, then poisoned)
                all_apples = [(pos, False) for pos in env.safe_apple_positions] + \
                             [(pos, True) for pos in env.poisoned_apple_positions]
                
                for i, (pos, is_poisoned) in enumerate(all_apples):
                    idx = 2 + 3*i
                    obs[idx] = pos[0]
                    obs[idx + 1] = pos[1]
                    obs[idx + 2] = 1.0 if is_poisoned else 0.0
            else:
                raise ValueError(f"Unknown observation type: {observation_type}")
            
            all_observations.append(obs)
            agent_positions.append(agent_pos)
    
    return np.array(all_observations), agent_positions

def generate_safe_actions_for_all_states(env, agent_positions, poisoned_apple_positions):
    """
    For each agent position, generate the list of safe actions (actions that don't lead to poisoned apples).
    
    Args:
        env: The environment
        agent_positions: List of (row, col) tuples for agent positions
        poisoned_apple_positions: List of (row, col) tuples for poisoned apples
    
    Returns:
        safe_actions_list: List of lists, where safe_actions_list[i] contains safe actions for agent_positions[i]
    """
    # Action definitions
    UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
    all_actions = [UP, RIGHT, DOWN, LEFT]
    action_names = ["UP", "RIGHT", "DOWN", "LEFT"]
    
    poisoned_set = set(poisoned_apple_positions)
    safe_actions_list = []
    
    for agent_pos in agent_positions:
        safe_actions = []
        
        for action in all_actions:
            # Calculate new position based on action
            new_pos = list(agent_pos)
            
            if action == UP:
                new_pos[0] = max(0, new_pos[0] - 1)
            elif action == RIGHT:
                new_pos[1] = min(env.grid_size - 1, new_pos[1] + 1)
            elif action == DOWN:
                new_pos[0] = min(env.grid_size - 1, new_pos[0] + 1)
            elif action == LEFT:
                new_pos[1] = max(0, new_pos[1] - 1)
            
            # Check if new position is safe (not a poisoned apple)
            if tuple(new_pos) not in poisoned_set:
                safe_actions.append(action)
        
        safe_actions_list.append(safe_actions)
    
    return safe_actions_list