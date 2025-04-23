import gymnasium as gym
from stable_baselines3 import PPO  # 或者您使用的其他算法，例如 A2C, SAC, TD3 等
import numpy as np
import argparse
import os

def evaluate_car_racing(model_path, episodes=10, seed=42, render=False, deterministic=True):
    """
    使用 Stable Baselines3 加载并评估 CarRacing 模型。

    Args:
        model_path (str): 已训练模型的路径 (.zip 文件).
        episodes (int): 评估的回合数.
        seed (int): 环境和模型随机种子.
        render (bool): 是否渲染环境.
        deterministic (bool): 是否使用确定性动作选择.
    """
    print(f"开始评估模型: {model_path}")
    print(f"评估回合数: {episodes}")

    if not os.path.exists(model_path):
        print(f"错误: 模型文件未找到: {model_path}")
        return

    # 创建环境
    # 使用连续动作空间 (continuous=True), 这是 CarRacing-v2 的默认设置之一
    # Stable Baselines3 通常会自动处理图像输入，如果模型是用 SB3 训练的
    render_mode = "human" if render else None
    env = gym.make("CarRacing-v3", continuous=True, render_mode=render_mode, domain_randomize=False) # 使用连续动作空间

    # 设置随机种子 (如果需要，尽管 SB3 加载时通常会处理)
    # np.random.seed(seed)
    # env.reset(seed=seed)

    try:
        # 加载模型 (确保算法类与训练时使用的匹配)
        # 您可能需要更改 PPO 为您实际使用的算法 (e.g., A2C, SAC)
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
        obs, info = env.reset(seed=seed + episode) # 为每个回合设置不同的种子
        episode_reward = 0.0
        terminated = False
        truncated = False
        step = 0

        while not terminated and not truncated:
            # 从模型获取动作
            action, _states = model.predict(obs, deterministic=deterministic)

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            step += 1

            if render:
                # CarRacing 不需要手动调用 render()，它在 make 时通过 render_mode 控制
                pass

            # 可选：添加一个最大步数以防万一 (尽管 CarRacing 有自己的机制)
            # if step > 2000:
            #    print(f"回合 {episode + 1}: 达到自定义最大步数，截断。")
            #    truncated = True

        total_rewards.append(episode_reward)
        total_steps.append(step)
        print(f"回合 {episode + 1}/{episodes}: 总奖励 = {episode_reward:.2f}, 步数 = {step}")
        # 可以添加更详细的结束原因判断，如果 info 字典提供的话
        # print(f"  结束状态: terminated={terminated}, truncated={truncated}, info={info}")


    env.close()
    print("评估完成。")

    # 计算并打印平均奖励和步数
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
