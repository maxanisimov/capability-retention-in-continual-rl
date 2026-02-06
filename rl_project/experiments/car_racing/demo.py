#%%
import gymnasium
import numpy as np
import sys
sys.path.append('/Users/ma5923/Documents/_projects/CertifiedContinualLearning/rl_project')
sys.path.append('/Users/ma5923/Documents/_projects/CertifiedContinualLearning')
from utils.ppo_utils import ppo_train, PPOConfig
from src.trainer import IntervalTrainer
from utils.gymnasium_utils import render_gymnasium_agent

class StateBasedWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # Features: car position, velocity, angle, angular velocity, wheel angles, wheel velocities
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32
        )
    
    def observation(self, obs):
        # Access car state from the environment
        car = self.unwrapped.car
        
        features = [
            car.hull.position.x / 100.0,  # Normalized position
            car.hull.position.y / 100.0,
            car.hull.angle,
            car.hull.linearVelocity.x / 10.0,
            car.hull.linearVelocity.y / 10.0,
            car.hull.angularVelocity,
            car.wheels[0].joint.angle,  # Front wheel angle
            car.wheels[1].joint.angle,  # Rear wheel angle
            car.wheels[0].omega,        # Front wheel angular velocity
            car.wheels[1].omega,        # Rear wheel angular velocity
        ]
        
        return np.array(features, dtype=np.float32)


class CarRacingSafetyWrapper(gymnasium.Wrapper):
    """
    Safety wrapper for CarRacing environment.
    
    Tracks safety based on:
    - On track vs off track (grass)
    - Speed limits
    - Extreme angles
    """
    
    def __init__(self, env, max_speed=50.0, max_angle=0.8):
        super().__init__(env)
        self.max_speed = max_speed
        self.max_angle = max_angle
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['safe'] = True
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Get car state
        car = self.unwrapped.car
        
        # Check speed safety
        speed = np.sqrt(
            car.hull.linearVelocity.x**2 + 
            car.hull.linearVelocity.y**2
        )
        speed_safe = speed < self.max_speed
        
        # Check angle safety (not tilted too much)
        angle_safe = abs(car.hull.angle) < self.max_angle
        
        # Check if on track (grass penalty in reward indicates off-track)
        # CarRacing gives -0.1 reward per frame on grass
        on_track = reward > -0.1  # Heuristic: if reward is very negative, likely on grass
        
        # Overall safety
        is_safe = speed_safe and angle_safe and on_track
        
        info['safe'] = is_safe
        info['speed'] = speed
        info['angle'] = car.hull.angle
        
        return obs, reward, terminated, truncated, info

#%%
# Usage
env1 = gymnasium.make('CarRacing-v3', render_mode='human')
env1 = StateBasedWrapper(env1)
env1 = CarRacingSafetyWrapper(env1, max_speed=50.0, max_angle=0.8)

ppo_cfg = PPOConfig(
    total_timesteps=100_000,
    lr=3e-4,
    minibatch_size=64,
    # n_epochs=10,
    gamma=0.99,
    # lam=0.95,
    # clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5
)
actor, critic, training_data = ppo_train(env=env1, cfg=ppo_cfg, return_training_data=True)

#%%