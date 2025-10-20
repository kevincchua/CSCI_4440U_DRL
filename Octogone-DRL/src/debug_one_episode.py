import sys
sys.path.append('.')
from envs.octogone_env import OctogoneEnv
import numpy as np

def debug_one_episode(**kwargs):
    """
    Run one episode with random actions and print step-by-step details.
    """
    env = OctogoneEnv(render_mode='human', **kwargs)
    obs, info = env.reset()
    env.print_level_layout()
    done = False
    step = 0

    print(f"Step {step}:")
    print(f"  Position: {obs[:2]}")
    print(f"  Velocity: {obs[2:4]}")
    print(f"  Momentum: {obs[5]}")
    print("-" * 20)

    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        env.render()
        step += 1

        print(f"Step {step}:")
        print(f"  Action: {action}")
        print(f"  Position: {obs[:2]}")
        print(f"  Velocity: {obs[2:4]}")
        print(f"  Momentum: {obs[5]}")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")
        print("-" * 20)

    env.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_platforms', type=int, default=10, help='Number of platforms in the level')
    parser.add_argument('--num_spikes', type=int, default=5, help='Number of spikes in the level')
    parser.add_argument('--max_gap_size', type=int, default=200, help='Maximum gap size between platforms')
    args = parser.parse_args()

    env_kwargs = {
        'num_platforms': args.num_platforms,
        'num_spikes': args.num_spikes,
        'max_gap_size': args.max_gap_size
    }

    debug_one_episode(**env_kwargs)
