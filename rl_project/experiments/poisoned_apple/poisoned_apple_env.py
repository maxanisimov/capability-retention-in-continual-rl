"""
Poisoned Apple Environment - A Gymnasium-based grid world where an agent
must collect safe apples while avoiding poisoned ones.

The environment features:
- Grid-based navigation
- Multiple apples (some may be poisoned)
- Limited steps (equal to number of apples)
- Rewards for collecting safe apples, penalties for poisoned ones
"""

import numpy as np
import gymnasium
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

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
        self.max_steps = max_steps if max_steps is not None else num_apples**2 # TODO: remove max_steps limit
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
            # "unsafe": tuple(self.agent_pos) in self.poisoned_apples
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
