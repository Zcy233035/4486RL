import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import argparse
import os

def evaluate_car_racing(model_path, episodes=10, seed=42, render=False, deterministic=True):
    
    print(f"开始评估模型: {model_path}")
    print(f"评估回合数: {episodes}")

    if not os.path.exists(model_path):
        print(f"错误: 模型文件未找到: {model_path}")
        return

    
    
    render_mode = "human" if render else None
    env = gym.make("CarRacing-v3", continuous=True, render_mode=render_mode, domain_randomize=False)

    try:
        model = PPO.load(model_path, env=env)
        print("模型加载成功。")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        env.close()
        return

    total_rewards = []
    total_steps = []
    print("开始运行评估...")

    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        episode_reward = 0.0
        terminated = False
        truncated = False
        step = 0
        while not terminated and not truncated:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            if render:
                pass
        total_rewards.append(episode_reward)
        total_steps.append(step)
        print(f"回合 {episode + 1}/{episodes}: 总奖励 = {episode_reward:.2f}, 步数 = {step}")
        
    env.close()
    print("评估完成。")
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    avg_steps = np.mean(total_steps)
    std_steps = np.std(total_steps)

    print(f"--- 评估结果 ({episodes} 回合) ---")
    print(f"平均奖励: {avg_reward:.2f} +/- {std_reward:.2f}")
    print(f"平均步数: {avg_steps:.2f} +/- {std_steps:.2f}")

    return avg_reward, std_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估 CarRacing Stable Baselines3 模型")
    parser.add_argument("--model-path", type=str, default="trained_models/ppo_carRacing .zip", help="已训练模型的路径 (.zip 文件, 默认为 trained_models/ppo_carRacing .zip)")
    parser.add_argument("--episodes", type=int, default=100, help="评估回合数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--render", action="store_true", help="是否渲染环境")
    args = parser.parse_args()

    evaluate_car_racing(args.model_path, args.episodes, args.seed, args.render)
