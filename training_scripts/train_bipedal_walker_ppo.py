import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import os

def main():
    device = "cpu"

    env_id = "BipedalWalker-v3"
    vec_env = make_vec_env(env_id, n_envs=4)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log="./ppo_bipedalwalker_tensorboard/",
        device=device,
        n_steps=2048,
        batch_size=64,
    )

    print("开始训练...")
    total_train_steps = 3_000_000
    model.learn(total_timesteps=total_train_steps, tb_log_name="first_run")
    print("训练完成！")

    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)
    model_path_base = os.path.join(save_dir, "ppo_bipedalwalker")
    model.save(model_path_base)
    print(f"模型已保存到: {model_path_base}.zip")

    print("开始评估...")
    model_zip_path = model_path_base + ".zip"

    eval_env = gym.make(env_id)
    loaded_model = PPO.load(model_zip_path, env=eval_env, device=device)

    mean_reward, std_reward = evaluate_policy(loaded_model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"评估结果: 平均奖励 = {mean_reward:.2f} +/- {std_reward:.2f}")

    eval_env.close()
    vec_env.close()

if __name__ == "__main__":
    main()