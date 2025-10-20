import yaml
import os
import pandas as pd
import sys
sys.path.append(os.getcwd())
from stable_baselines3 import PPO, A2C
from envs.octogone_env import OctogoneEnv
import gymnasium as gym
import numpy as np

MAX_STEPS_PER_EPISODE = 5000

def evaluate(config_file, model_path, num_episodes=10, render=False, **kwargs):
    """
    Evaluate a trained model.
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Create the environment
    render_mode = "human" if render else None
    env = OctogoneEnv(reward_persona=config['env']['reward_persona'], render_mode=render_mode, **kwargs)

    # Load the model
    if config['model']['name'] == 'PPO':
        model = PPO.load(model_path)
    elif config['model']['name'] == 'A2C':
        model = A2C.load(model_path)
    else:
        raise ValueError(f"Unknown model: {config['model']['name']}")

    # Evaluate the model
    all_metrics = []

    for i in range(num_episodes):
        obs, info = env.reset()
        done = False
        num_steps = 0

        episode_metrics = {
            'episode': i,
            'reward': 0,
            'length': 0,
            'time_to_goal': np.nan,
            'death': 0,
            'unique_tiles_visited': set(),
            'max_x_velocity': 0,
            'max_y_velocity': 0,
            'max_momentum': 0
        }

        while not done and num_steps < MAX_STEPS_PER_EPISODE:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            if render:
                env.render()

            episode_metrics['reward'] += reward
            episode_metrics['length'] += 1
            episode_metrics['unique_tiles_visited'].add(tuple(np.round(obs[:2]).astype(int)))
            episode_metrics['max_x_velocity'] = max(episode_metrics['max_x_velocity'], abs(obs[2]))
            episode_metrics['max_y_velocity'] = max(episode_metrics['max_y_velocity'], abs(obs[3]))
            episode_metrics['max_momentum'] = max(episode_metrics['max_momentum'], obs[5])

            num_steps += 1

        if done:
            episode_metrics['time_to_goal'] = env.elapsed_time
            if env.player_pos[1] < -100:
                episode_metrics['death'] = 1

        all_metrics.append(episode_metrics)

    env.close()

    # Process and save metrics
    df = pd.DataFrame(all_metrics)
    df['unique_tiles_visited'] = df['unique_tiles_visited'].apply(len)

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    os.makedirs('logs', exist_ok=True)
    df.to_csv(f"logs/{model_name}_evaluation_metrics.csv", index=False)

    aggregate_metrics = {
        'mean_reward': float(df['reward'].mean()),
        'std_reward': float(df['reward'].std()),
        'mean_length': float(df['length'].mean()),
        'mean_time_to_goal': float(df['time_to_goal'].mean()),
        'total_deaths': int(df['death'].sum()),
        'mean_unique_tiles': float(df['unique_tiles_visited'].mean()),
        'mean_max_x_velocity': float(df['max_x_velocity'].mean()),
        'mean_max_y_velocity': float(df['max_y_velocity'].mean()),
        'mean_max_momentum': float(df['max_momentum'].mean())
    }

    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    with open(f"logs/{model_name}_aggregate_metrics.json", 'w') as f:
        import json
        json.dump(aggregate_metrics, f, indent=4)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--model', type=str, required=True, help='Path to the model file')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--num_platforms', type=int, default=10, help='Number of platforms in the level')
    parser.add_argument('--num_spikes', type=int, default=5, help='Number of spikes in the level')
    parser.add_argument('--max_gap_size', type=int, default=200, help='Maximum gap size between platforms')
    args = parser.parse_args()

    env_kwargs = {
        'num_platforms': args.num_platforms,
        'num_spikes': args.num_spikes,
        'max_gap_size': args.max_gap_size
    }

    evaluate(args.config, args.model, args.episodes, args.render, **env_kwargs)
