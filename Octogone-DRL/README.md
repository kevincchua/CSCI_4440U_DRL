# Octogone DRL Testing Framework

This repository contains the source code for the video game "Octogone" and a comprehensive Deep Reinforcement Learning (DRL) testing framework. This document focuses on the DRL framework.


### Installation

It is recommended to use a Python virtual environment to manage the project's dependencies.

1.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Project Structure

*   `configs/`: Contains YAML files for configuring training runs.
*   `envs/`: Contains the `OctogoneEnv` environment.
*   `src/`: Contains the training, evaluation, and debugging scripts.
*   `models/`: (Git-ignored) Directory where trained models are saved.
*   `logs/`: (Git-ignored) Directory where evaluation results are saved.
*   `octogone_files/`: Contains the original Godot game files.

## Training an Agent

To train an agent, you use the `src/train.py` script. The training process is controlled by a configuration file from the `configs/` directory.

```
python3 src/train.py --config configs/ppo_speedrunner.yaml
```

This command will train a PPO agent with the "speedrunner" reward persona for 100,000 timesteps. The trained model will be saved as `models/ppo_speedrunner.zip`.

## Evaluating an Agent

To evaluate a trained agent, you use the `src/evaluate.py` script.

```
python3 src/evaluate.py --config configs/ppo_speedrunner.yaml --model models/ppo_speedrunner.zip --episodes 20
```

This will run 20 episodes of evaluation and save the results to the `logs/` directory.

### Optional Arguments

    `--render`: Use this flag to watch the agent play a very simplified version.
    ```bash
    python3 src/evaluate.py --config configs/ppo_speedrunner.yaml --model models/ppo_speedrunner.zip --episodes 5 --render
    ```



# Octogone
Octogone is a 2D platformer that is entirely focused around the conservation of momentum and speedrunning.

The game is about Octo, a failed lab experiment, who is escaping his lab using some weird guns they found.

Use the shotgun and grappling hook to blast yourself to victory!

Controls:
- Movement: A and D or left- and right-arrow
- Jumping: Spacebar or up-arrow
- Aiming: Mouse movement

The game currently features 12 levels and is easily edited to create more.


# Technical Stuff
This GitHub repository contains the video game Octogone and its code.

The game has been made in the Godot game engine, so to ensure a smooth editing feel, open the project.godot file in Godot.

A similar warning has been provided at the top of the aforementioned file.

The newest stable download for Godot can be found on https://godotengine.org/

Also this code is in no ways even close to being perfect and we are simply two highschool students working on it, but of course criticism is allowed and can be provided with a simple bug report.

We highly encoure you to do so if you find anything noteworthy!

# Credits
A huge thank you to:

- The Godot Community for providing us with a great game engine and good documentation.
- The Totem Game Dev student team of the TU/e for providing us with the casestudy and the freedom to let us make what we thought would be a fun and engaging game.
- ExceptRea on itch.io for making a big part of the tileset we used for our levels.
- Lesiakower on pixabay for composing the absolute bop of this game: "Lab Vibes"
- And lastly a huge thank you to all of our friends who helped building and playtesting the game to ensure that all the physics and levels were on par!

Sincerely, Team RND:

- Lead Programmer: Finn
- Level Design: Cogi
- Prototyping and QA: Pepijn
- Art Direction: Philip

# DRL Testing Framework

This repository also contains a Deep Reinforcement Learning (DRL) testing framework for Octogone. This framework allows you to train and evaluate DRL agents to test the game's mechanics and design.

## How to Run

### 1. Setup the Environment

First, you need to install the required Python packages. It is recommended to use a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the Models

You can train the DRL models using the `src/train.py` script. The training process is configured using YAML files located in the `configs/` directory.

To train a specific model, run the following command:

```bash
python3 src/train.py --config configs/<config_file.yaml>
```

For example, to train the PPO model with the "Speedrunner" persona, run:

```bash
python3 src/train.py --config configs/ppo_speedrunner.yaml
```

The trained models will be saved in the `models/` directory.

### 3. Evaluate the Models

After training, you can evaluate the performance of the models using the `src/evaluate.py` script.

To evaluate a model, run the following command:

```bash
python3 src/evaluate.py --config configs/<config_file.yaml> --model models/<model_file.zip> --episodes <num_episodes>
```

For example, to evaluate the PPO "Speedrunner" model for 10 episodes, run:

```bash
python3 src/evaluate.py --config configs/ppo_speedrunner.yaml --model models/ppo_speedrunner.zip --episodes 10
```

To run the evaluation with visualization, add the `--render` flag:

```bash
python3 src/evaluate.py --config configs/ppo_speedrunner.yaml --model models/ppo_speedrunner.zip --episodes 10 --render
```

You can also configure the procedural level generation for the evaluation:

```bash
python3 src/evaluate.py --config configs/ppo_speedrunner.yaml --model models/ppo_speedrunner.zip --episodes 10 --num_platforms 20 --num_spikes 10 --max_gap_size 300
```

The evaluation results, including per-episode metrics and aggregate statistics, will be saved in the `logs/` directory.

### 4. Debug the Environment

A debugging script is provided to run a single episode with random actions and print step-by-step details of the environment.

```bash
python3 src/debug_one_episode.py
```

You can also configure the procedural level generation for the debugging script:

```bash
python3 src/debug_one_episode.py --num_platforms 5 --num_spikes 2 --max_gap_size 100
```

This is useful for visualizing the environment's behavior and ensuring the physics are working as expected.
