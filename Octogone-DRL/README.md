# Octogone DRL Testing Framework

This repository contains the source code for the video game "Octogone" and a comprehensive Deep Reinforcement Learning (DRL) testing framework. 

## 1. Overview

The DRL framework is a powerful tool for testing the game mechanics, agent behaviors, and level design of Octogone. It is built using Python, `gymnasium`, and `stable-baselines3`. 


### 2.1. Installation

It is recommended to use a Python virtual environment to manage the project's dependencies.

1.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    venv/scripts/activate
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### 2.2. Project Structure

*   `configs/`: Contains YAML files for configuring training runs.
*   `envs/`: Contains the `OctogoneEnv` environment.
*   `src/`: Contains the training and evaluation.
*   `models/`: Directory where trained models are saved (Git-ignored)
*   `logs/`: Directory where evaluation results are saved (Git-ignored)
*   `octogone_files/`: Contains the original Godot game files.

## 3. Training an Agent

To train an agent, you use the `src/train.py` script. 

```bash
python src/train.py --config configs/ppo_speedrunner.yaml
```

This command will train a PPO agent with the "speedrunner" reward persona for 100,000 timesteps. The trained model will be saved as `models/ppo_speedrunner.zip`.

## 4. Evaluating an Agent

To evaluate a trained agent, you use the `src/evaluate.py` script.

```bash
python src/evaluate.py --config configs/ppo_speedrunner.yaml --model models/ppo_speedrunner.zip --episodes 20
```

This will run 20 episodes of evaluation and save the results to the `logs/` directory.

### Rendering

*   `--render`: Use this flag to watch the agent play in a `pygame` window.
    ```bash
    python src/evaluate.py --config configs/ppo_speedrunner.yaml --model models/ppo_speedrunner.zip --episodes 5 --render
    ```

## Analytics

The `analytics/generate_plots.py` script generates plots from the evaluation data.

```bash
python analytics/generate_plots.py
```
