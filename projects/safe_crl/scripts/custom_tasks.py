# custom_tasks.py
# Three lightweight Gymnasium environments:
#  - DrunkSpider-v0      : stochastic gridworld reach-avoid (safety-friendly)
#  - MinitaurSimple-v0   : toy quadruped gait task (no external physics)
#  - CubeRotate-v0       : quaternion rotation-to-target task
#
# Gymnasium API compatible (>=0.28). No third-party deps beyond numpy.
# Author: (you)

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------------------------
# Utilities
# ---------------------------

def _np_random(seed: Optional[int]):
    # Gymnasium-compatible RNG creation
    # Returns (rng, used_seed)
    return gym.utils.seeding.np_random(seed)

def _quat_mul(q1, q2):
    # Hamilton product, q = [w, x, y, z]
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    out = np.stack([w, x, y, z], axis=-1)
    return out

def _quat_normalize(q):
    norm = np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8
    return q / norm

def _quat_from_axis_angle(axis, angle):
    # axis: (3,), angle in radians
    axis = np.asarray(axis, dtype=np.float32)
    n = np.linalg.norm(axis) + 1e-8
    a = axis / n
    s = math.sin(angle / 2.0)
    return np.array([math.cos(angle / 2.0), a[0]*s, a[1]*s, a[2]*s], dtype=np.float32)

def _quat_angle(q1, q2):
    # Smallest angle between orientations q1 and q2 (both normalized)
    # cos(theta/2) = |<q1, q2>|
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot)

# ============================================================
# 1) DrunkSpider-v0
# ============================================================

@dataclass
class DrunkSpiderConfig:
    width: int = 10
    height: int = 10
    max_steps: int = 200
    slip_prob: float = 0.2         # probability to replace intended move with a random neighbor
    lava_density: float = 0.10     # fraction of cells that are lava (excludes start/goal)
    step_penalty: float = -0.01
    goal_reward: float = 1.0
    lava_penalty: float = -1.0
    render_cell_px: int = 24       # for rgb_array render

class DrunkSpiderEnv(gym.Env):
    """
    Stochastic reach-avoid grid. 8-direction movements, with 'drunk' slip.
    Observation: concatenated one-hots: [agent_onehot, goal_onehot, lava_mask] -> R^(3*H*W)
    Action: Discrete(8) (N, NE, E, SE, S, SW, W, NW)
    Terminate on reaching goal or stepping on lava. 'info["is_failure"]=True' on lava.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 6}

    def __init__(self, cfg: DrunkSpiderConfig = DrunkSpiderConfig(), goal=None, lava=None, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.render_mode = render_mode
        self.H, self.W = cfg.height, cfg.width
        n = self.H * self.W
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3 * n,), dtype=np.float32)
        self.action_space = spaces.Discrete(8)
        self._dirs = np.array([
            (-1,  0),  # N
            (-1, +1),  # NE
            ( 0, +1),  # E
            (+1, +1),  # SE
            (+1,  0),  # S
            (+1, -1),  # SW
            ( 0, -1),  # W
            (-1, -1),  # NW
        ], dtype=np.int32)
        self.np_random, _ = _np_random(None)
        self._reset_maps(goal=goal, lava=lava)

    def _reset_maps(self, goal=None, lava=None):
        # Set start in first column, middle row and goal in last column, middle row
        middle_row = self.H // 2
        self.start = (middle_row, 0)
        self.goal = goal if goal is not None else (middle_row, self.W - 1)
        
        if lava is not None:
            assert lava.shape == (self.H, self.W)
            self._lava = lava
        else:
            # Create horizontal lava belts that force path through narrow bridge
            lava = np.zeros((self.H, self.W), dtype=bool)
            
            # Define the bridge corridor (middle row where start and goal are)
            bridge_row = middle_row
            
            # Create two horizontal lava belts above and below the bridge
            # Upper lava belt
            upper_belt_row = bridge_row - 1
            lower_belt_row = bridge_row + 1
            
            # Create horizontal lava belts across most of the width
            # Leave gaps at the edges and a narrow bridge passage in the middle
            bridge_start_col = 2
            bridge_end_col = self.W - 2
            
            for c in range(self.W):
                # Upper lava belt
                if 0 <= upper_belt_row < self.H:
                    # Create belt with bridge opening in the middle section
                    if (bridge_start_col <= c <= bridge_end_col):
                        lava[upper_belt_row, c] = True
                    # Also add lava above the belt to make it thicker
                    if 0 <= upper_belt_row - 1 < self.H:
                        if (bridge_start_col <= c <= bridge_end_col):
                            lava[upper_belt_row - 1, c] = True
                
                # Lower lava belt
                if 0 <= lower_belt_row < self.H:
                    # Create belt with bridge opening in the middle section
                    if (bridge_start_col <= c <= bridge_end_col):
                        lava[lower_belt_row, c] = True
                    # Also add lava below the belt to make it thicker
                    if 0 <= lower_belt_row + 1 < self.H:
                        if (bridge_start_col <= c <= bridge_end_col):
                            lava[lower_belt_row + 1, c] = True
            
            # Add some strategic lava in the bridge corridor to make navigation challenging
            # but ensure there's always a path
            for c in range(bridge_start_col + 1, bridge_end_col):
                if self.np_random.random() < self.cfg.lava_density * 0.3:
                    # Add lava above or below the main path, but not on it
                    if self.np_random.random() < 0.5:
                        if bridge_row - 1 >= 0:
                            lava[bridge_row - 1, c] = True
                    else:
                        if bridge_row + 1 < self.H:
                            lava[bridge_row + 1, c] = True
            
            # Add additional scattered lava in non-critical areas
            for r in range(self.H):
                for c in range(self.W):
                    # Skip if already lava or on critical path
                    if lava[r, c] or (r, c) in (self.start, self.goal):
                        continue
                    
                    # Don't block the main horizontal corridor
                    if r == bridge_row:
                        continue
                        
                    # Add random lava with reduced density
                    if self.np_random.random() < self.cfg.lava_density * 0.4:
                        # Only if it doesn't block essential access paths
                        if not self._blocks_essential_path(r, c, bridge_row):
                            lava[r, c] = True
            
            # Ensure start and goal positions and their immediate neighbors are clear
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    # Clear around start
                    sr, sc = self.start[0] + dr, self.start[1] + dc
                    if 0 <= sr < self.H and 0 <= sc < self.W:
                        lava[sr, sc] = False
                    # Clear around goal
                    gr, gc = self.goal[0] + dr, self.goal[1] + dc
                    if 0 <= gr < self.H and 0 <= gc < self.W:
                        lava[gr, gc] = False
            
            # Ensure the main path (bridge row) is navigable
            lava[bridge_row, :] = False
            
            self._lava = lava
        self._agent = list(self.start)
        self._t = 0

    def _blocks_essential_path(self, r, c, bridge_row):
        """Check if placing lava at (r,c) would block essential access to the bridge."""
        # Protect vertical access to the bridge from start and goal areas
        start_row, start_col = self.start
        goal_row, goal_col = self.goal
        
        # Protect columns near start and goal for vertical access
        if (c <= 2 and abs(r - start_row) <= 2) or \
           (c >= self.W - 3 and abs(r - goal_row) <= 2):
            return True
            
        # Protect direct vertical access to bridge corridor
        if abs(r - bridge_row) == 1 and (c <= 2 or c >= self.W - 3):
            return True
            
        return False

    def _onehot(self, r, c):
        idx = r * self.W + c
        vec = np.zeros((self.H * self.W,), dtype=np.float32)
        vec[idx] = 1.0
        return vec

    def _obs(self):
        agent = self._onehot(self._agent[0], self._agent[1])
        goal = self._onehot(self.goal[0], self.goal[1])
        lava = self._lava.astype(np.float32).reshape(-1)
        return np.concatenate([agent, goal, lava], axis=0).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.np_random, _ = _np_random(seed)
        self._reset_maps()
        return self._obs(), {"is_failure": False}

    def step(self, action: int):
        self._t += 1
        assert self.action_space.contains(action)
        # slip
        
        if self.np_random.random() < self.cfg.slip_prob:
            action = int(self.np_random.integers(0, 8))
        dr, dc = self._dirs[action]
        nr = int(np.clip(self._agent[0] + dr, 0, self.H - 1))
        nc = int(np.clip(self._agent[1] + dc, 0, self.W - 1))
        self._agent = [nr, nc]

        terminated = False
        truncated = self._t >= self.cfg.max_steps
        reward = self.cfg.step_penalty
        info = {"is_failure": False}

        if self._lava[nr, nc]:
            reward += self.cfg.lava_penalty
            terminated = True
            info["is_failure"] = True
        elif (nr, nc) == self.goal:
            reward += self.cfg.goal_reward
            terminated = True

        return self._obs(), float(reward), bool(terminated), bool(truncated and not terminated), info

    def render(self):
        assert self.render_mode == "rgb_array", "Only rgb_array supported"
        px = self.cfg.render_cell_px
        img = np.zeros((self.H * px, self.W * px, 3), dtype=np.uint8)
        # colors
        col_bg = np.array([240, 240, 240], dtype=np.uint8)
        col_lava = np.array([220, 50, 32], dtype=np.uint8)
        col_goal = np.array([30, 160, 30], dtype=np.uint8)
        col_agent = np.array([40, 90, 200], dtype=np.uint8)
        img[:] = col_bg
        for r in range(self.H):
            for c in range(self.W):
                y0, y1 = r * px, (r + 1) * px
                x0, x1 = c * px, (c + 1) * px
                if self._lava[r, c]:
                    img[y0:y1, x0:x1] = col_lava
        # goal and agent
        r, c = self.goal
        img[r*px:(r+1)*px, c*px:(c+1)*px] = col_goal
        r, c = self._agent
        img[r*px:(r+1)*px, c*px:(c+1)*px] = col_agent
        return img

# ============================================================
# 2) MinitaurSimple-v0 (toy quadruped)
# ============================================================

@dataclass
class MinitaurSimpleConfig:
    dt: float = 0.04             # control timestep
    max_steps: int = 600
    # joint bounds (radians)
    hip_limit: float = math.radians(60)
    knee_limit: float = math.radians(90)
    # action is joint angular accelerations scaled by:
    accel_limit: float = math.radians(600)
    # reward weights
    w_vel: float = 1.0
    w_ctrl: float = 0.005
    w_pose: float = 0.002
    w_stability: float = 0.05
    target_speed: float = 1.0     # m/s proxy target

class MinitaurSimpleEnv(gym.Env):
    """
    Toy quadruped with 8 joints [hip,knee] x 4 legs. No physics engine; instead:
    - state: joint angles & velocities (16), forward x and x_dot (2) => 18-D
    - contact heuristic: a leg is "in contact" if its knee is flexed and hip near neutral
    - forward velocity proxy grows with alternating leg velocity while in contact
    Reward: + (x_dot towards +X) - control - pose penalty - instability penalty
    """
    metadata = {"render_modes": []}

    def __init__(self, cfg: MinitaurSimpleConfig = MinitaurSimpleConfig(), render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.render_mode = render_mode
        high_angles = np.array(
            [cfg.hip_limit, cfg.knee_limit] * 4 + [np.inf]*2, dtype=np.float32
        )
        high_vel = np.array(
            [np.inf, np.inf] * 4 + [np.inf]*2, dtype=np.float32
        )
        # observation: [angles(8), vels(8), x, x_dot] -> 18
        self.observation_space = spaces.Box(
            low=-np.concatenate([high_angles[:8], np.full(8, np.inf, dtype=np.float32), np.array([np.inf, np.inf], dtype=np.float32)]),
            high=np.concatenate([high_angles[:8], np.full(8, np.inf, dtype=np.float32), np.array([np.inf, np.inf], dtype=np.float32)]),
            dtype=np.float32
        )
        # action: 8 accelerations in [-1,1] scaled to accel_limit
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.np_random, _ = _np_random(None)
        self._t = 0
        self._angles = None
        self._vels = None
        self._x = 0.0
        self._x_dot = 0.0

    def _obs(self):
        return np.concatenate([self._angles, self._vels, [self._x, self._x_dot]]).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.np_random, _ = _np_random(seed)
        self._t = 0
        # small random init near standing
        self._angles = np.zeros(8, dtype=np.float32)
        self._vels = np.zeros(8, dtype=np.float32)
        self._x = 0.0
        self._x_dot = 0.0
        # jitter
        self._angles += self.np_random.normal(0.0, 0.05, size=8).astype(np.float32)
        return self._obs(), {}

    def step(self, action: np.ndarray):
        self._t += 1
        a = np.clip(action, -1.0, 1.0).astype(np.float32) * self.cfg.accel_limit
        dt = self.cfg.dt

        # integrate joint velocities/angles
        self._vels += a * dt
        # damp velocities a bit
        self._vels *= 0.98
        self._angles += self._vels * dt

        # clip joint limits
        # indices: [hip0,knee0, hip1,knee1, hip2,knee2, hip3,knee3]
        for i in range(4):
            hip_i, knee_i = 2*i, 2*i+1
            self._angles[hip_i] = float(np.clip(self._angles[hip_i], -self.cfg.hip_limit, self.cfg.hip_limit))
            self._angles[knee_i] = float(np.clip(self._angles[knee_i], 0.0, self.cfg.knee_limit))  # knees flex (0..limit)
            # zero vel if we hit limits (naive)
            if abs(self._angles[hip_i]) >= self.cfg.hip_limit - 1e-6: self._vels[hip_i] = 0.0
            if self._angles[knee_i] <= 1e-6 or self._angles[knee_i] >= self.cfg.knee_limit - 1e-6: self._vels[knee_i] = 0.0

        # contact heuristic: knee flexed (> 20deg) AND |hip| < 25deg
        contact = np.zeros(4, dtype=np.float32)
        for i in range(4):
            hip = self._angles[2*i]
            knee = self._angles[2*i+1]
            if (knee > math.radians(20)) and (abs(hip) < math.radians(25)):
                contact[i] = 1.0

        # forward velocity proxy:
        # encourage alternating leg motions: left side (0,2) vs right side (1,3)
        left_vel = (self._vels[0] - self._vels[2]) * 0.5
        right_vel = (self._vels[4] - self._vels[6]) * 0.5
        propulsive = (left_vel * (contact[0] + contact[2]) + right_vel * (contact[1] + contact[3]))
        # smooth & clamp
        self._x_dot = float(np.clip(0.3 * propulsive, -3.0, 3.0))
        self._x += self._x_dot * dt

        # pose cost: penalize extreme angles
        pose_cost = float(np.mean(self._angles**2))
        # control cost
        ctrl_cost = float(np.mean((action.astype(np.float32))**2))
        # instability penalty: many steps without any contact
        stability_pen = 1.0 if contact.sum() == 0.0 else 0.0

        # reward: velocity towards +X, shaped to target_speed
        speed_err = (self._x_dot - self.cfg.target_speed)
        r_vel = -abs(speed_err) + 1.0  # max 1 when exactly at target
        reward = (
            self.cfg.w_vel * r_vel
            - self.cfg.w_ctrl * ctrl_cost
            - self.cfg.w_pose * pose_cost
            - self.cfg.w_stability * stability_pen
        )

        terminated = False  # no terminal state; can add failure if desired
        truncated = self._t >= self.cfg.max_steps
        info = {
            "x": self._x,
            "x_dot": self._x_dot,
            "contact": contact,
        }
        return self._obs(), float(reward), bool(terminated), bool(truncated), info

# ============================================================
# 3) CubeRotate-v0
# ============================================================

@dataclass
class CubeRotateConfig:
    dt: float = 0.05
    max_steps: int = 200
    max_ang_vel: float = math.radians(60)  # rad/s per axis
    success_tol_deg: float = 5.0
    # reward
    success_bonus: float = 5.0
    w_err: float = 1.0
    w_ctrl: float = 0.01

class CubeRotateEnv(gym.Env):
    """
    Rotate a cube to a random target orientation.
    State: [q_current(4), q_target(4)]  (unit quaternions, wxyz)
    Action: angular velocity vector in R^3 (rad/s), Box([-1,1]^3) scaled to max_ang_vel
    Dynamics: q_{t+1} = normalize( dq(omega*dt) ⊗ q_t )
    Reward: - angle_error (radians) - w_ctrl * ||u||^2 + success_bonus if error < tol
    """
    metadata = {"render_modes": []}

    def __init__(self, cfg: CubeRotateConfig = CubeRotateConfig(), render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.render_mode = render_mode
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.np_random, _ = _np_random(None)
        self._t = 0
        self.q = None
        self.q_target = None

    def _sample_quat(self):
        # Uniform random rotation via random axis & angle
        axis = self.np_random.normal(0.0, 1.0, size=3).astype(np.float32)
        axis /= (np.linalg.norm(axis) + 1e-8)
        angle = float(self.np_random.uniform(0.0, math.pi))
        return _quat_from_axis_angle(axis, angle)

    def _obs(self):
        return np.concatenate([self.q, self.q_target]).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.np_random, _ = _np_random(seed)
        self._t = 0
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # identity
        self.q_target = self._sample_quat()
        return self._obs(), {}

    def step(self, action: np.ndarray):
        self._t += 1
        u = np.clip(action, -1.0, 1.0).astype(np.float32) * self.cfg.max_ang_vel
        dt = self.cfg.dt

        # integrate quaternion by angular velocity (axis-angle for small dt)
        ang = float(np.linalg.norm(u) * dt)
        if ang > 1e-8:
            dq = _quat_from_axis_angle(u, ang)
            self.q = _quat_normalize(_quat_mul(dq, self.q))
        # compute error angle
        qc = self.q
        qt = self.q_target
        err = float(_quat_angle(qc, qt))
        reward = - self.cfg.w_err * err - self.cfg.w_ctrl * float(np.dot(action, action))

        success = err <= math.radians(self.cfg.success_tol_deg)
        terminated = bool(success)
        if success:
            reward += self.cfg.success_bonus
        truncated = self._t >= self.cfg.max_steps
        info = {"angle_error_rad": err}
        return self._obs(), float(reward), bool(terminated), bool(truncated), info

# ============================================================
# Registration helpers
# ============================================================

def register_envs():
    gym.registry = getattr(gym, "registry", None)  # for gymnasium>=0.28, register() is in gymnasium.envs.registration
    from gymnasium.envs.registration import register

    register(
        id="DrunkSpider-v0",
        entry_point="custom_tasks:DrunkSpiderEnv",
        kwargs={"cfg": DrunkSpiderConfig()},
    )
    register(
        id="MinitaurSimple-v0",
        entry_point="custom_tasks:MinitaurSimpleEnv",
        kwargs={"cfg": MinitaurSimpleConfig()},
    )
    register(
        id="CubeRotate-v0",
        entry_point="custom_tasks:CubeRotateEnv",
        kwargs={"cfg": CubeRotateConfig()},
    )

# If the module is run directly, perform registration to allow manual testing.
if __name__ == "__main__":
    register_envs()
    # quick smoke tests
    for env_id in ["DrunkSpider-v0", "MinitaurSimple-v0", "CubeRotate-v0"]:
        env = gym.make(env_id)
        obs, info = env.reset(seed=0)
        total = 0.0
        for _ in range(5):
            a = env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            total += r
            if term or trunc:
                obs, info = env.reset()
        print(env_id, "ok, example return over 5 steps:", total)
        env.close()
