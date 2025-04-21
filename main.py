import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import os

def main():
    
    # --- 1. 创建环境 ---
    # BipedalWalker-v3 是环境的 ID
    # make_vec_env 会创建多个并行环境，加速数据收集 (n_envs=4 表示创建4个)
    # 如果你想先只用一个环境观察，可以设置 n_envs=1
    env_id = "BipedalWalker-v3"
    vec_env = make_vec_env(env_id, n_envs=4)

    # --- 2. 定义模型 ---
    # "MlpPolicy": 告诉 SB3 使用默认的 MLP 网络作为 Actor 和 Critic 的 backbone。
    #              SB3 会自动根据环境的观测空间和动作空间创建合适的 MLP 结构。
    # vec_env:     传递创建好的（向量化）环境。
    # verbose=1:   在训练时打印一些日志信息。
    # 其他超参数 (如 learning_rate, n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range)
    # 会使用 SB3 为 PPO 设定的默认值，这些值通常对很多环境都有效。
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./ppo_bipedalwalker_tensorboard/")

    # --- 3. 训练模型 ---
    # total_timesteps: 你想让 Agent 训练的总步数。
    #                  对于 BipedalWalker，100万步 (1e6) 是一个不错的起点，可能需要更多才能达到好的效果。
    #                  你可以先设小一点（比如 10000 或 50000）来快速测试流程是否跑通。
    # tb_log_name:   TensorBoard 日志的名称，你可以在终端用 tensorboard --logdir ./ppo_bipedalwalker_tensorboard/ 查看训练曲线。
    print("开始训练...")
    total_train_steps = 1_000_00 # 训练一百万步
    model.learn(total_timesteps=total_train_steps, tb_log_name="first_run")
    print("训练完成！")

    # --- 4. 保存模型 ---
    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)
    # 定义模型的基础路径名 (不含扩展名)
    model_path_base = os.path.join(save_dir, "ppo_bipedalwalker")
    # 保存时 SB3 会自动添加 .zip
    model.save(model_path_base)
    print(f"模型已保存到: {model_path_base}.zip")

    # --- 5. (可选) 评估训练好的模型 ---
    print("开始评估...")
    # 加载模型时，需要提供完整的 .zip 文件路径
    model_zip_path = model_path_base + ".zip"

    # 或者创建一个单独的环境来评估，并可以加上渲染
    eval_env = gym.make(env_id, render_mode="human") # 设置 render_mode="human" 可以看到画面
    # 使用包含 .zip 的路径加载模型
    loaded_model = PPO.load(model_zip_path, env=eval_env)

    # evaluate_policy 会运行模型 N 个 episode，然后返回平均奖励和标准差
    # n_eval_episodes: 评估的回合数
    # deterministic=True: 让 Actor 网络选择概率最高的动作，而不是采样，通常评估时用 True
    mean_reward, std_reward = evaluate_policy(loaded_model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"评估结果: 平均奖励 = {mean_reward:.2f} +/- {std_reward:.2f}")

    # 让模型跑一个 episode 看看效果
    print("运行一个 episode 进行演示...")
    obs, _ = eval_env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _states = loaded_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        episode_reward += reward
        # 注意：你可能需要在循环内部调用 render() 才能看到实时画面
        # eval_env.render() # 取消这行的注释以启用渲染
        if terminated or truncated:
            done = True
    print(f"演示 episode 的总奖励: {episode_reward}")

    eval_env.close()
    vec_env.close() # 关闭环境

if __name__ == "__main__":
    main()