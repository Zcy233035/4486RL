import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np


env = gym.make("BipedalWalkerHardcore-v3", render_mode="human")

model_path = "trained_models/ppo_bipedalwalkerHardcore .zip"

model = PPO.load(model_path, env=env)

vec_env = model.get_env()
obs = vec_env.reset()
total_reward = 0.0
num_episodes = 0
timesteps = 0
print_interval = 100

print("Starting evaluation...")
print("Timestep | Hull Angle | Hull AngVel | Vel X | Vel Y")
print("---------|------------|-------------|-------|-------")

while num_episodes < 1:
    action, _states = model.predict(obs, deterministic=True)
    current_obs_for_print = obs #

    new_obs, rewards, dones, info = vec_env.step(action)

    total_reward += rewards.item()
    timesteps += 1

    if timesteps % print_interval == 0:
        if isinstance(current_obs_for_print, np.ndarray) and current_obs_for_print.ndim == 2:
            param1 = current_obs_for_print[0, 0] 
            param2 = current_obs_for_print[0, 1] 
            param3 = current_obs_for_print[0, 2] 
            param4 = current_obs_for_print[0, 3] 
            print(f"{timesteps:>8} | {param1:>10.3f} | {param2:>11.3f} | {param3:>5.3f} | {param4:>5.3f}")
        else:
            print(f"Timestep {timesteps}: obs format unexpected.")

    obs = new_obs

    if dones.any():
        print("-" * 50)
        print(f"Episode finished after {timesteps} timesteps.")
        final_steps = info[0].get('episode', {}).get('l', timesteps)
        final_reward = info[0].get('episode', {}).get('r', total_reward)
        print(f"Total reward for the episode: {final_reward:.2f}")
        print(f"(Info from Monitor: steps={final_steps}, reward={final_reward:.2f})")
        if final_steps < 1600 and final_reward > 200:
             print("Termination reason: Likely reached the end.")
        elif final_steps < 1600:
             print("Termination reason: Likely fell.")
        else:
             print("Termination reason: Reached max timesteps.")
        num_episodes += 1

env.close()
