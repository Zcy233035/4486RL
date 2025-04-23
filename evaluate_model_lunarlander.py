import gymnasium as gym
from stable_baselines3 import PPO
import time

# --- Parameters --- #
ENV_NAME = "LunarLander-v3"
MODEL_PATH = "ppo_lunarlander_sb3.zip"
N_EVAL_EPISODES = 10
DETERMINISTIC_POLICY = True
# ------------------ #

if __name__ == "__main__":
    # Create the environment with rendering
    env = gym.make(ENV_NAME, render_mode="human")

    # Load the trained model
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
            # Rendering is handled by the env wrapper in human mode usually,
            # but explicit render call might be needed depending on setup/version
            # env.render()
            # time.sleep(0.01) # Optional delay

        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
        total_rewards.append(episode_reward)

    env.close()
    print("\nEvaluation finished.")

    # Calculate and print average reward
    if total_rewards:
        avg_reward = sum(total_rewards) / len(total_rewards)
        print(f"Average Reward over {N_EVAL_EPISODES} episodes: {avg_reward:.2f}")