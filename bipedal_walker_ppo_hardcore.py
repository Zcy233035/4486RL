import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
import os
import numpy as np

def main():
    device = "cpu"
    env_id = "BipedalWalkerHardcore-v3"
    n_envs = 16
    total_train_steps = 50000000
    save_dir = "trained_models"
    log_dir = "./ppo_bipedalwalkerHardcore_tensorboard/"
    model_save_name = "ppo_bipedalwalkerHardcore_normalized"
    stats_save_name = "vecnormalize_stats"

    vec_env = make_vec_env(env_id, n_envs=n_envs)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.)

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=log_dir,
        device=device,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=lambda fraction: 2.5e-4 * fraction,
        policy_kwargs=policy_kwargs,
    )

    print("开始训练...")
    model.learn(total_timesteps=total_train_steps, tb_log_name=model_save_name)
    print("训练完成！")

    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{model_save_name}.zip")
    stats_path = os.path.join(save_dir, f"{stats_save_name}.pkl")

    model.save(model_path)
    vec_env.save(stats_path)
    print(f"模型已保存到: {model_path}")
    print(f"环境统计数据已保存到: {stats_path}")

    print("开始评估...")
    eval_env_raw = make_vec_env(env_id, n_envs=1)
    eval_env = VecNormalize.load(stats_path, eval_env_raw)
    eval_env.training = False
    eval_env.norm_reward = False

    loaded_model = PPO.load(model_path, env=eval_env, device=device)

    mean_reward, std_reward = evaluate_policy(loaded_model, eval_env, n_eval_episodes=10, deterministic=True, warn=False)
    print(f"评估结果: 平均奖励 = {mean_reward:.2f} +/- {std_reward:.2f}")

    eval_env.close()

if __name__ == "__main__":
    main()