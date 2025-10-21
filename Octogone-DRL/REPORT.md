# Comprehensive Project Report: DRL Framework for Octogone

## 1. Project Overview

This document provides a detailed breakdown of the Deep Reinforcement Learning (DRL) framework created to test the game "Octogone." The framework is designed to be a flexible and powerful tool for analyzing game mechanics, testing agent behaviors, and identifying potential design flaws through the use of DRL agents.

The core of the project is a reimplementation of the Octogone game mechanics in a Python-based `gymnasium` environment. This allows for rapid, headless training and evaluation of DRL agents, which can be configured to adopt different "personas" (e.g., a speedrunner or a survivor) through reward shaping.

## 2. File-by-File Breakdown

### 2.1. `envs/octogone_env.py`

*   **Purpose:** This file contains the core of the project: the `OctogoneEnv` class, which is a `gymnasium` environment that simulates the physics and mechanics of the Octogone game.
*   **Role:** It serves as the bridge between the DRL agent and the game world, providing the agent with observations, executing its actions, and returning rewards.
*   **Key Components:**
    *   **`OctogoneEnv` class:**
        *   **`__init__(self, ...)`:** The constructor initializes the environment, sets up the physics constants, defines the observation and action spaces, and configures the procedural level generation. It also initializes `pygame` if the `render_mode` is set to `"human"`.
        *   **`_generate_level(self)`:** This method procedurally generates a new level layout on each call, creating a randomized arrangement of platforms, spikes, and a goal. The complexity of the level is determined by the `num_platforms`, `num_spikes`, and `max_gap_size` parameters.
        *   **`step(self, action)`:** This is the main physics loop of the environment. It takes an action from the agent, updates the player's position and velocity based on the game's physics, checks for collisions with platforms and spikes, and returns the new state, reward, and a `done` flag.
        *   **`reset(self, ...)`:** This method resets the environment to its initial state, which includes generating a new level, placing the player at the start, and resetting the elapsed time and other state variables.
        *   **`render(self)`:** If the `render_mode` is `"human"`, this method uses `pygame` to draw the current state of the game world to the screen.
        *   **`close(self)`:** This method cleanly closes the `pygame` window.
        *   **`print_level_layout(self)`:** This utility method prints a simple ASCII representation of the current level layout to the console, which is useful for debugging.

### 2.2. `src/train.py`

*   **Purpose:** This script is used to train the DRL agents.
*   **Role:** It reads a YAML configuration file, creates an `OctogoneEnv` instance and a Stable-Baselines3 model (PPO or A2C), and then starts the training process.
*   **Key Components:**
    *   **`train(config_file)` function:** This function encapsulates the entire training process. It parses the config file, initializes the environment and model, calls the `.learn()` method on the model, and saves the trained model to the `models/` directory.

### 2.3. `src/evaluate.py`

*   **Purpose:** This script is used to evaluate the performance of a trained DRL agent.
*   **Role:** It loads a trained model, runs a specified number of episodes in the environment, collects a comprehensive set of metrics, and saves the results to CSV and JSON files.
*   **Key Components:**
    *   **`evaluate(...)` function:** This function orchestrates the evaluation process. It loads the model, and then loops for the specified number of episodes. In each episode, it runs the agent until the episode ends (either by reaching the goal, dying, or hitting the `MAX_STEPS_PER_EPISODE` limit). It collects metrics for each episode, and then saves the collected data to the `logs/` directory.

### 2.4. `src/debug_one_episode.py`

*   **Purpose:** This script is a debugging tool for running a single episode of the environment with random actions.
*   **Role:** It allows for a step-by-step inspection of the environment's state, the actions being taken, and the rewards being received.
*   **Key Components:**
    *   **`debug_one_episode()` function:** This function creates an `OctogoneEnv` instance, and then runs a single episode, printing detailed information at each step.

### 2.5. `configs/*.yaml`

*   **Purpose:** These YAML files contain the configuration parameters for the training process.
*   **Role:** They allow for easy modification of hyperparameters without changing the source code.
*   **Parameters:**
    *   **`env`:**
        *   **`name`:** The name of the environment (e.g., "OctogoneEnv-v0").
        *   **`reward_persona`:** The reward persona to use ("speedrunner" or "survivor").
    *   **`model`:**
        *   **`name`:** The DRL algorithm to use ("PPO" or "A2C").
        *   **`policy`:** The type of policy network to use (e.g., "MlpPolicy").
        *   **`total_timesteps`:** The total number of training steps.
        *   Other parameters are algorithm-specific hyperparameters (e.g., `learning_rate`, `gamma`, etc.).

## 3. Project Workflow

### 3.1. Training Workflow

1.  **Configuration:** The user selects a YAML configuration file from the `configs/` directory.
2.  **Execution:** The user runs the `src/train.py` script, passing the path to the config file as a command-line argument.
3.  **Initialization:** The `train` function reads the config file, creates an `OctogoneEnv` instance, and initializes a Stable-Baselines3 model with the specified hyperparameters.
4.  **Training Loop:** The `.learn()` method of the model is called, which starts the main training loop. The agent interacts with the environment for `total_timesteps`, collecting experience and updating its policy.
5.  **Saving:** Once training is complete, the trained model is saved to a `.zip` file in the `models/` directory.

### 3.2. Evaluation Workflow

1.  **Execution:** The user runs the `src/evaluate.py` script, providing the paths to the config file and the trained model, and optionally the number of episodes and a `--render` flag.
2.  **Initialization:** The `evaluate` function loads the trained model and creates an `OctogoneEnv` instance.
3.  **Evaluation Loop:** The script loops for the specified number of episodes. In each episode, the agent interacts with the environment until the episode ends.
4.  **Metrics Collection:** During each episode, a variety of metrics are collected, including the total reward, episode length, time to goal, whether the agent died, the number of unique tiles visited, and the maximum momentum achieved.
5.  **Saving Results:** After all episodes are complete, the collected metrics are saved to a per-episode CSV file and an aggregate JSON file in the `logs/` directory.

## 4. Important Details

### 4.1. Reward Function

The reward function is crucial for shaping the agent's behavior. The `OctogoneEnv` supports two reward personas:
*   **`speedrunner`:** This persona rewards the agent for reaching the goal quickly. It receives a large positive reward for finishing and a small negative reward for each time step, encouraging efficiency.
*   **`survivor`:** This persona rewards the agent for staying alive and exploring. It receives a small positive reward for movement (proportional to its velocity) and a large negative penalty for dying.

### 4.2. Evaluation Metrics

*   **`reward`:** The total reward accumulated by the agent in an episode.
*   **`length`:** The total number of steps taken in an episode.
*   **`time_to_goal`:** The in-game time elapsed when the agent reaches the goal.
*   **`death`:** A boolean flag indicating whether the agent died in the episode.
*   **`unique_tiles_visited`:** The number of unique grid cells the agent visited, which is a measure of exploration.
*   **`max_x_velocity`, `max_y_velocity`, `max_momentum`:** These metrics track the maximum horizontal velocity, vertical velocity, and overall momentum (magnitude of the velocity vector) achieved by the agent in an episode.

### 4.3. Procedural Generation and Variation

The environment's procedural generation is the key to preventing agents from simply memorizing a single level layout. By generating a new level on every `reset()`, the agent is forced to learn a more generalizable policy that can adapt to different arrangements of platforms, gaps, and spikes. The configurability of the level generation allows for fine-tuning the difficulty and variety of the levels, which is essential for robust training and testing.
