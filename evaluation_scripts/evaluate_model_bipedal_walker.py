import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

# 加载环境
# 将 render_mode 设置为 None 来禁用渲染
env = gym.make("BipedalWalker-v3", render_mode=None)

# 指定模型路径
model_path = "trained_models/ppo_bipedalwalker.zip"

# 加载模型
model = PPO.load(model_path, env=env)

# 运行模型进行评估
vec_env = model.get_env()
obs = vec_env.reset()
episode_rewards = [] # 存储每次评估的回报
num_episodes = 100 # 要评估的总次数
current_episode = 0
total_reward = 0.0
timesteps = 0


print(f"Starting evaluation for {num_episodes} episodes...")

# 修改循环条件以评估 100 次
while current_episode < num_episodes:
    action, _states = model.predict(obs, deterministic=True)
    # 移除 current_obs_for_print = obs

    # 执行动作
    new_obs, rewards, dones, info = vec_env.step(action)

    total_reward += rewards.item()
    timesteps += 1

    # 移除状态打印逻辑
    # if timesteps % print_interval == 0: ...

    obs = new_obs

    if dones.any():
        # 记录本次评估的回报
        final_reward = info[0].get('episode', {}).get('r', total_reward)
        episode_rewards.append(final_reward)
        print(f"Episode {current_episode + 1}/{num_episodes} finished. Reward: {final_reward:.2f}")

        # 重置用于下一次评估
        total_reward = 0.0
        timesteps = 0
        # obs = vec_env.reset() # Monitor wrapper 会自动重置环境，这里不需要手动重置
        current_episode += 1
        # 移除详细的结束原因打印

# 计算并打印平均回报
mean_reward = np.mean(episode_rewards)
std_reward = np.std(episode_rewards)
print("\n" + "=" * 50)
print(f"Evaluation finished over {num_episodes} episodes.")
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
print("=" * 50)

env.close()