#%%
import gymnasium
import numpy as np
import sys
sys.path.append('/Users/ma5923/Documents/_projects/CertifiedContinualLearning/rl_project')
sys.path.append('/Users/ma5923/Documents/_projects/CertifiedContinualLearning')
from utils.ppo_utils import PPOConfig, ppo_train, evaluate

def one_hot_encode_state(state, num_states, task_num: int = 0) -> np.ndarray:
    """Convert discrete state to one-hot encoding with task indicator appended."""
    # There are 500 states + 1 task indicator
    encoded = np.zeros(num_states + 1, dtype=np.float32)
    encoded[state] = 1.0
    # encoded[-1] = task_num
    return encoded

class TaxiSafetyWrapper(gymnasium.Wrapper):
    """Marks illegal pickup/dropoff actions as unsafe via info['safe']."""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['safe'] = not (action in (4, 5) and reward == -10)
        return obs, reward, terminated, truncated, info

class OneHotWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env, task_num: int):
        super().__init__(env)
        low_bounds = np.zeros(env.observation_space.n + 1, dtype=np.float32)
        high_bounds = np.ones(env.observation_space.n + 1, dtype=np.float32)
        high_bounds[-1] = np.inf  # Last dimension (task indicator) is unbounded above
        self.observation_space = gymnasium.spaces.Box(
            low=low_bounds, high=high_bounds, dtype=np.float32
        )
        self.task_num = task_num
    
    def observation(self, obs):
        return one_hot_encode_state(obs, self.env.observation_space.n, self.task_num)

#%%
env = gymnasium.make('Taxi-v3')
env = TaxiSafetyWrapper(env)
env = OneHotWrapper(env, task_num=0)
obs, ifo = env.reset(seed=42)

#%%
ppo_config = PPOConfig(seed=42, total_timesteps=200_000)
actor, critic, training_data = ppo_train(env, ppo_config, return_training_data=True)

#%%
avg_reward, std_reward, failure_rate = evaluate(actor=actor, env=env, episodes=10)

#%%
from utils.gymnasium_utils import render_gymnasium_agent
env_render = gymnasium.make('Taxi-v3', render_mode='human')
env_render = TaxiSafetyWrapper(env_render)
env_render = OneHotWrapper(env_render, task_num=0)
render_gymnasium_agent(
    env=env_render,
    actor=actor, 
    num_episodes=1, 
    seed=42
)
env_render.close()
# %%
