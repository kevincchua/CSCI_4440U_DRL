# Octogone DRL Testing Framework

This repository contains the source code for the video game "Octogone" and a comprehensive Deep Reinforcement Learning (DRL) testing framework. This document focuses on the DRL framework.

## 1. Overview

The DRL framework is a powerful tool for testing the game mechanics, agent behaviors, and level design of Octogone. It is built using Python, `gymnasium`, and `stable-baselines3`. The core of the framework is a custom `gymnasium` environment that reimplements the game's physics and allows for procedural level generation with curriculum learning.

## 2. Getting Started

### 2.1. Installation

It is highly recommended to use a Python virtual environment to manage the project's dependencies.

1.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### 2.2. Project Structure

*   `configs/`: Contains YAML files for configuring training runs.
*   `envs/`: Contains the `OctogoneEnv` environment.
*   `src/`: Contains the training, evaluation, and debugging scripts.
*   `models/`: (Git-ignored) Directory where trained models are saved.
*   `logs/`: (Git-ignored) Directory where evaluation results are saved.
*   `octogone_files/`: Contains the original Godot game files.

## 3. Training an Agent with Curriculum Learning

To train an agent, you use the `src/train.py` script. The training process now includes a curriculum learning callback that gradually increases the difficulty of the generated levels.

**Example Command:**
```bash
python src/train.py --config configs/ppo_speedrunner.yaml
```

This command will train a PPO agent with the "speedrunner" reward persona for 100,000 timesteps, with the difficulty of the environment increasing at predefined intervals. The trained model will be saved as `models/ppo_speedrunner.zip`.

## 4. Evaluating an Agent

To evaluate a trained agent, you use the `src/evaluate.py` script.

**Example Command:**
```bash
python src/evaluate.py --config configs/ppo_speedrunner.yaml --model models/ppo_speedrunner.zip --episodes 20
```

This will run 20 episodes of evaluation and save the results to the `logs/` directory.

### 4.1. Optional Arguments

*   `--render`: Use this flag to watch the agent play in a `pygame` window.
    ```bash
    python src/evaluate.py --config configs/ppo_speedrunner.yaml --model models/ppo_speedrunner.zip --episodes 5 --render
    ```

*   **Level Generation Parameters:** You can control the complexity of the procedurally generated levels:
    *   `--num_platforms`: The number of platforms to generate.
    *   `--num_spikes`: The number of spikes to generate.
    *   `--max_gap_size`: The maximum gap size between platforms.
    ```bash
    python src/evaluate.py --config configs/ppo_speedrunner.yaml --model models/ppo_speedrunner.zip --episodes 10 --num_platforms 15 --num_spikes 7
    ```

## 5. Debugging

The `src/debug_one_episode.py` script is a useful tool for inspecting the environment's behavior. It runs a single episode with random actions and prints detailed information at each step.

**Example Command:**
```bash
python src/debug_one_episode.py
```

You can also use the level generation parameters with the debugging script:
```bash
python src/debug_one_episode.py --num_platforms 5 --max_gap_size 100
```

## 6. Troubleshooting

*   **Identical Evaluation Results:** If your evaluation results are identical across episodes, it is likely because the environment is not being reset with a new random seed. The current implementation ensures a new level is generated on each `reset()`. If you need more randomness, you can pass a different `--seed` to the environment's constructor.
*   **Rendering Fails:** The visualization is based on `pygame`. If you encounter rendering errors, ensure that you have a display server running (e.g., X11 on Linux) and that your Python environment has the necessary graphics libraries installed. The ALSA errors related to the audio system can be safely ignored.

For a complete, in-depth explanation of the project's architecture, workflow, and components, please refer to the `REPORT.md` file.
