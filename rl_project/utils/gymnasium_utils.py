import gymnasium
import torch
import numpy as np

ENV_ENGINES = ['gymnasium'] # , 'safety_gymnasium']

def render_gymnasium_agent(
        actor, 
        env: gymnasium.Env | None = None,
        env_id: str | None = None, 
        env_engine: str = 'gymnasium',
        num_episodes: int = 1,
        log_std: float | None = None, deterministic: bool = True,
        seed: int = 42, 
    ):
    """
    Render the agent in the Gymnasium or Safety Gymnasium environment using human render mode.

    Args:
        actor: Trained policy network
        env: Gymnasium environment instance (if None, env_id must be provided)
        env_id: Environment ID (to create environment if env is None)
        env_engine: 'gymnasium' or 'safety_gymnasium' (used if env_id is provided)
        num_episodes: Number of episodes to render
        log_std: Log standard deviation for continuous action spaces
        deterministic: Whether to use deterministic actions
        seed: Random seed
    
    Returns:
        None
    """
    assert (env is not None) or (env_id is not None), "Either env or env_id must be provided"

    if env is not None:
        assert env.unwrapped.render_mode == 'human', "Environment must be created with render_mode='human'"
        env_id = env.spec.id
    elif env_id is not None:
        assert env_engine in ENV_ENGINES, f"env_engine must be one of {ENV_ENGINES}"
        if env_engine == 'gymnasium':
            env = gymnasium.make(
                env_id,
                render_mode='human',
            )
        else:
            raise NotImplementedError("Safety Gymnasium rendering not implemented yet")
            # env = safety_gymnasium.make(
            #     env_id,
            #     render_mode='human',
            # )
            # env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)

    if not deterministic and isinstance(env.action_space, gymnasium.spaces.Box):
        assert log_std is not None, "log_std must be provided for stochastic continuous actions"
        assert log_std > 0, "log_std must be positive"

    print(f"\n\n=== Rendering {actor} in {env_id} ===")

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")

        done = False
        obs, _ = env.reset(seed=seed + episode)
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                if isinstance(env.action_space, gymnasium.spaces.Box):
                    # Continuous actions: use mean from actor
                    if deterministic:
                        action = actor(obs_t).cpu().numpy()[0]
                    else:
                        mean = actor(obs_t)
                        std = np.exp(log_std).expand_as(mean)
                        dist = torch.distributions.Normal(mean, std)
                        action = dist.sample().cpu().numpy()[0]
                    # Clip to action space bounds
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                else:
                    # Discrete actions: use argmax
                    if deterministic:
                        logits = actor(obs_t)
                        action = torch.argmax(logits, dim=-1).item()
                    else:
                        logits = actor(obs_t)
                        dist = torch.distributions.Categorical(logits=logits)
                        action = dist.sample().item()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    env.close()
    