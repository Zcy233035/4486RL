import gymnasium as gym
from stable_baselines3 import PPO
import time

ENV_NAME = "LunarLander-v3"
MODEL_PATH = "trained_models/ppo_lunarlander_sb3.zip"
N_EVAL_EPISODES = 1000
DETERMINISTIC_POLICY = True

if __name__ == "__main__":
    env = gym.make(ENV_NAME, render_mode="human")

    model = PPO.load(MODEL_PATH, env=env)

    total_rewards = []
    print(f"Running evaluation for {N_EVAL_EPISODES} episodes...")

    for episode in range(N_EVAL_EPISODES):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0

        while not terminated and not truncated:
            action, _states = model.predict(obs, deterministic=DETERMINISTIC_POLICY)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
        total_rewards.append(episode_reward)

    env.close()
    print("\nEvaluation finished.")

    if total_rewards:
        avg_reward = sum(total_rewards) / len(total_rewards)
        print(f"Average Reward over {N_EVAL_EPISODES} episodes: {avg_reward:.2f}")