import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


max_training_timesteps = 3000000


if __name__ == '__main__':
    print("============================================================================================")

    env_name = "LunarLander-v3"
    
    n_envs = 8
    env = make_vec_env(env_name, n_envs=n_envs)
    

    
    
    model = PPO("MlpPolicy", env, verbose=1, 
                learning_rate=0.0003, 
                n_steps=2048,      
                batch_size=64,      
                n_epochs=10,       
                gamma=0.99,         
                gae_lambda=0.95,   
                clip_range=0.2,     
                ent_coef=0.0,       
                vf_coef=0.5,        
                max_grad_norm=0.5,  
                tensorboard_log="./ppo_lunarlander_tensorboard/"
                )

    print("Starting training with Stable Baselines3 PPO...")
    
    model.learn(total_timesteps=max_training_timesteps, progress_bar=True)

    print("Training finished.")

    
    model_save_path = "ppo_lunarlander_sb3"
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}.zip")

    
    print("Evaluating the trained agent...")
    
    eval_env = gym.make(env_name)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Evaluation results: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()
    eval_env.close() 
