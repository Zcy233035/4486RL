import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

env = gym.make("BipedalWalker-v3", render_mode=None)
model_path = "trained_models/ppo_bipedalwalker.zip"
model = PPO.load(model_path, env=env)

vec_env = model.get_env()
obs = vec_env.reset()
episode_rewards = [] 
num_episodes = 100 
current_episode = 0
total_reward = 0.0
timesteps = 0


print(f"Starting evaluation for {num_episodes} episodes...")


while current_episode < num_episodes:
    action, _states = model.predict(obs, deterministic=True)
    new_obs, rewards, dones, info = vec_env.step(action)
    total_reward += rewards.item()
    timesteps += 1
    obs = new_obs
    if dones.any():
        
        final_reward = info[0].get('episode', {}).get('r', total_reward)
        episode_rewards.append(final_reward)
        print(f"Episode {current_episode + 1}/{num_episodes} finished. Reward: {final_reward:.2f}")

        
        total_reward = 0.0
        timesteps = 0
        
        current_episode += 1
        
mean_reward = np.mean(episode_rewards)
std_reward = np.std(episode_rewards)
print("\n" + "=" * 50)
print(f"Evaluation finished over {num_episodes} episodes.")
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
print("=" * 50)

env.close()