#%%
import gymnasium
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO

env = gymnasium.make("ALE/Pacman-v5", obs_type='ram')

model = PPO("MlpPolicy", env, verbose=1)

# Retrieve the model from the hub
## repo_id =  id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name})
## filename = name of the model zip file from the repository
checkpoint = load_from_hub(
	repo_id="alfredowh/ppo-Pacman-v5",
	filename="ppo-ALE-Pacman-v5.zip",
)
PPO.load(checkpoint)

obs, info = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
    if done:
      obs, info = env.reset()

env.close()

#%%