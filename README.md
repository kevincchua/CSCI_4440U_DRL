# CSCI_4440U Deep Reinforcement Learning Assignment Repo

This repository contains implementations, environments, source code, analytics, and results for **Deep Reinforcement Learning (DRL)** coursework for CSCI 4440U. It features **multiple RL game environments and test frameworks** including custom Asteroids and Octogone-DRL agents.

---

## Table of Contents

- Overview
- Project Structure
- Key Environments
  - Asteroids RL Environment
  - Octogone-DRL Environment
- Folders and Contents
- Installation and Setup
- .gitignore Details
- Contributors

---

## Overview

This repo is designed for research, experimentation, and benchmarking of RL agents in complex, procedurally generated environments using modern RL libraries. It includes two full RL environments with code, configurations, analytics, and documentation for testing different RL agent archetypes.

---

## Project Structure

**Top-level folders:**
- `Asteroids/` — Asteroids RL environment & experiments
- `Octogone-DRL/` — Octogone platformer RL environment & experiments
- `.gitignore` — large config to keep repo clean
- `README.md` — short summary and reward table

**Asteroids/ Subfolders:**
- `.vscode/` — IDE configs
- `code/` — Environment, training scripts, evaluation, helpers
- `models/` — Pretrained models, checkpoints (git-ignored)
- `mylogs/` — TensorBoard logs, evaluation CSVs/PNGs
- `notebooks/` — Jupyter exploratory analysis and reports
- `menu.py` — Custom UI/menu logic
- `requirements.txt` — Project dependencies
- SFX assets (`chime.wav`)

**Octogone-DRL/ Subfolders:**
- `analysis/` — Analytics scripts and graph outputs
- `analytics/` — CSV summary statistics, visualizations
- `configs/` — YAML configuration files
- `envs/` — Custom RL environments (OctogoneEnv)
- `logs/` — Training and evaluation logs (git-ignored)
- `octogone_files/` — Original Godot platformer code/assets
- `src/` — Main entry points for training/evaluation
- `tests/` — Environment unit tests
- `requirements.txt` — dependencies

---

## Key Environments

### Asteroids RL Environment

A classic Asteroids-style game with three RL archetypes (“personas”):
- **Survivor:** Rewards for staying alive, defensive play
- **Hunter:** Rewards for offense (destroying asteroids), accuracy bonus
- **Speedrunner:** Rewards for rapid level completion, time pressure

Each persona has a **custom reward structure**.
Full source in `Asteroids/code/`, supporting training and evaluation with analytics in `mylogs/`.

### Octogone-DRL Environment

A procedurally generated platformer implemented as a Gymnasium-compatible environment (`OctogoneEnv`). Supports curriculum learning, evaluation, analytics, visualization, and various configurations via YAML.
All code and analytics in `Octogone-DRL/`.

---

## Folder and File Details

### Asteroids

- **code/**: Main RL environment, agent logic, reward calculations, helper modules.
- **models/**: Saved models and checkpoints (excluded from git for size/noise).
- **mylogs/**: TensorBoard logs, evaluation results, graphs.
- **notebooks/**: Exploratory analyses, agent evaluations, etc.

### Octogone-DRL

- **analysis/** & **analytics/**: Graph generation scripts, episode analytics in CSV/JSON.
- **configs/**: All YAML agent and environment configs (easy hyperparameter sweeps).
- **envs/**: Custom environment for procedural platformer RL (fully modular).
- **logs/**: Training/evaluation results.
- **octogone_files/**: Godot game engine source for visualization and editing.
- **src/**: Train/evaluate pipeline scripts.
- **tests/**: Unit tests for reliability.
- **requirements.txt**: Dependency list.

### .gitignore

A very thorough Python `.gitignore` covering:
- Model checkpoints, logs, cache (`outputs`, `Asteroids/models/checkpoints/`, `tb_logs/`)
- PyInstaller, pip, poetry, pdm, marimo, cursor configs
- Any common IDE, build, env, cache, test, coverage, lock files
- Large extracts from community Python templates for broad compatibility

---

## Installation and Setup

Each environment (`Asteroids`, `Octogone-DRL`) has its own requirements.txt. 

General steps:

**For Asteroids-DRL**

`cd Asteroids`

`python -m venv venv`

`source venv/bin/activate`

`python menu.py` or `pip install -r requirements.txt`

To run the project run the menu options.

**For Octogone-DRL**

`cd Octogone-DRL`

`python -m venv venv`

`source venv/bin/activate`

`venv/scripts/activate`

`pip install -r requirements.txt`

## To train an agent, you use the `src/train.py` script. 

```bash
python src/train.py --config configs/ppo_speedrunner.yaml
```

This command will train a PPO agent with the "speedrunner" reward persona for 100,000 timesteps. The trained model will be saved as `models/ppo_speedrunner.zip`.

## To evaluate a trained agent, you use the `src/evaluate.py` script.

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
python3 analytics/generate_plots.py
```

---

## Contributors

- [kevincchua](https://github.com/kevincchua)
- [Muneeb312](https://github.com/Muneeb312)

---

