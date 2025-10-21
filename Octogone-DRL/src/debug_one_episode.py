import gymnasium as gym
import numpy as np
import argparse
import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.octogone_env import OctogoneEnv


def debug_one_episode(env):
    """
    Runs one episode with random actions and prints detailed step-by-step information.
    """
    obs, info = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    print("=" * 20)
    print("Starting new episode...")
    print("=" * 20)

    env.print_level_layout()

    while not done:
        action = env.action_space.sample()  # Sample a random action

        print(f"Step {step_count}:")
        print(f"  - Player Pos Before: {env.player_pos}")
        print(f"  - Player Vel Before: {env.player_vel}")
        print(f"  - Action: {action}")

        obs, reward, done, _, info = env.step(action)

        print(f"  - Player Pos After: {env.player_pos}")
        print(f"  - Player Vel After: {env.player_vel}")
        print(f"  - Reward: {reward}")
        print(f"  - Done: {done}")
        print(f"  - Is on floor: {env.is_on_floor}")
        print("-" * 20)

        total_reward += reward
        step_count += 1

        if step_count > 5000:  # Safety break
            print("Episode exceeded 5000 steps. Terminating.")
            break

    print("=" * 20)
    print("Episode finished.")
    print(f"Total Reward: {total_reward}")
    print(f"Total Steps: {step_count}")
    print("=" * 20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--difficulty",
        type=int,
        default=0,
        help="Set the difficulty level for the environment.",
    )
    args = parser.parse_args()

    env = OctogoneEnv(difficulty=args.difficulty)
    debug_one_episode(env)
    env.close()
