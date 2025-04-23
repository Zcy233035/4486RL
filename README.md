# Reinforcement Learning for Box2D Environments

This project focuses on training Proximal Policy Optimization (PPO) agents for various continuous control environments within the Box2D physics engine, using the Gymnasium and Stable Baselines3 libraries.

## Project Structure

```
.
├── box2d/               # (env)
├── evaluation_scripts/  # Scripts to evaluate trained models
├── pic/                 # Images used in the report
├── tensorboard_results/ # Stores TensorBoard log files for experiments
├── trained_models/      # Stores trained agent models (.zip files)
├── training_scripts/    # Scripts to train agents for different environments
├── README.md            # This file
├── report.md            # Project report in Markdown format
├── report.pdf           # Generated PDF version of the report
└── requirements.txt     # Python dependencies
```

## Dependencies

The required Python libraries are listed in the `requirements.txt` file. You will need Python 3.8+.

Key dependencies include:
*   `gymnasium[box2d]`: For the simulation environments.
*   `stable-baselines3[extra]`: For the PPO implementation and utilities.
*   `torch`: As the backend for Stable Baselines3.
*   `tensorboard`: For visualizing training logs.


## Installation

1.  Clone the repository.
2.  Navigate to the project directory in your terminal.
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```


## Training

Training scripts are located in the `training_scripts/` directory. Each script trains a PPO agent for a specific environment.

*   **Environments Covered**: `CarRacing-v3`, `BipedalWalker-v3`, `LunarLander-v3` (potentially others like `BipedalWalkerHardcore-v3` if scripts exist).
*   **Algorithm**: Proximal Policy Optimization (PPO) from Stable Baselines3.
*   **Policies**: `CnnPolicy` for image-based input (e.g., `CarRacing-v3`), `MlpPolicy` for vector-based input (e.g., `BipedalWalker-v3`, `LunarLander-v3`).
*   **Hyperparameters**: Defined within each specific training script (e.g., `training_scripts/train_car_rancing_ppo.py`).
*   **Logging**: TensorBoard logs are saved to the `tensorboard_results/` directory. You can view them by running `tensorboard --logdir tensorboard_results/` in your terminal.

## Evaluation

Evaluation scripts are located in the `evaluation_scripts/` directory. These scripts load a pre-trained model from `trained_models/` and evaluate its performance on the corresponding environment.

Some training scripts may also include a basic evaluation step after training completes.

## Output

*   **Trained Models**: Saved as `.zip` files in the `trained_models/` directory.
*   **TensorBoard Logs**: Saved in the `tensorboard_results/` directory, organized by experiment runs.
*   **Report**: A detailed report is available as `report.md` and `report.pdf`.

