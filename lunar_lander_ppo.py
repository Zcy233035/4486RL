import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# Hyperparameters (many are handled by SB3 defaults, but can be customized)
# learning_rate = 0.0003 # SB3 default for PPO
# gamma = 0.99
# gae_lambda = 0.95 # SB3 default
# clip_range = 0.2 # SB3 default
# n_epochs = 10 # SB3 default
# n_steps = 2048 # SB3 default batch size (similar to T_horizon)
max_training_timesteps = 3000000 # Max timesteps to train

# Removed RolloutBuffer, ActorCritic, and custom PPO class definitions
# ... existing code ...


if __name__ == '__main__':
    print("============================================================================================")

    env_name = "LunarLander-v3"
    # Using make_vec_env is recommended for SB3, especially for parallel training
    n_envs = 8 # Define the number of environments
    env = make_vec_env(env_name, n_envs=n_envs)
    # env = gym.make(env_name) # Keep single env creation commented out or remove

    # SB3 PPO Model - Using default hyperparameters unless specified
    # Check SB3 documentation for all available hyperparameters
    model = PPO("MlpPolicy", env, verbose=1, # policy_kwargs=dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])]), # Example customization
                learning_rate=0.0003, # Default is 0.0003
                n_steps=2048,      # Default is 2048
                batch_size=64,      # Default is 64
                n_epochs=10,       # Default is 10
                gamma=0.99,         # Default is 0.99
                gae_lambda=0.95,   # Default is 0.95
                clip_range=0.2,     # Default is 0.2
                ent_coef=0.0,       # Default is 0.0
                vf_coef=0.5,        # Default is 0.5
                max_grad_norm=0.5,  # Default is 0.5
                tensorboard_log="./ppo_lunarlander_tensorboard/"
                )

    print("Starting training with Stable Baselines3 PPO...")
    # Train the agent
    model.learn(total_timesteps=max_training_timesteps, progress_bar=True)

    print("Training finished.")

    # Save the trained model
    model_save_path = "ppo_lunarlander_sb3"
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}.zip")

    # Evaluate the trained agent (optional)
    print("Evaluating the trained agent...")
    # You might need a separate evaluation environment if using VecEnv for training
    eval_env = gym.make(env_name)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Evaluation results: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()
    eval_env.close() # Close evaluation env as well

    # Example of loading and using the trained model
    # print("\nLoading the saved model...")
    # loaded_model = PPO.load(model_save_path)
    # print("Running the loaded model for a few episodes...")
    # obs, _ = eval_env.reset()
    # for _ in range(500):
    #     action, _states = loaded_model.predict(obs, deterministic=True)
    #     obs, rewards, terminated, truncated, info = eval_env.step(action)
    #     if terminated or truncated:
    #         obs, _ = eval_env.reset()
    # eval_env.close()
