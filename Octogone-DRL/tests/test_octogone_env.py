import pytest
import numpy as np
import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.octogone_env import OctogoneEnv


@pytest.fixture
def env():
    """Fixture to create a default OctogoneEnv instance."""
    return OctogoneEnv(seed=42)


def test_env_creation(env):
    """Tests if the environment is created successfully."""
    assert env is not None
    assert env.action_space.n == 3
    assert env.observation_space.shape == (6,)


def test_env_reset(env):
    """Tests the reset method of the environment."""
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (6,)
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)
    # Check that player position is reset (start_pos is random but should be consistent with seed)
    assert env.player_pos[0] != 0 or env.player_pos[1] != 0


def test_env_step(env):
    """Tests the step method of the environment."""
    env.reset()
    action = env.action_space.sample()  # Sample a random action
    obs, reward, done, truncated, info = env.step(action)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (6,)
    assert env.observation_space.contains(obs)

    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_reproducibility_with_seed():
    """Tests if the environment produces the same level with the same seed."""
    env1 = OctogoneEnv(seed=123, difficulty=1)
    env2 = OctogoneEnv(seed=123, difficulty=1)

    obs1, _ = env1.reset()
    obs2, _ = env2.reset()

    # The initial observation should be identical
    np.testing.assert_array_equal(obs1, obs2)

    # The generated level layout should be identical
    platforms1 = [p["rect"] for p in env1.level["platforms"]]
    platforms2 = [p["rect"] for p in env2.level["platforms"]]
    assert platforms1 == platforms2

    spikes1 = [s["rect"] for s in env1.level["spikes"]]
    spikes2 = [s["rect"] for s in env2.level["spikes"]]
    assert spikes1 == spikes2

    np.testing.assert_array_equal(env1.level["start_pos"], env2.level["start_pos"])
    np.testing.assert_array_equal(env1.level["goal_pos"], env2.level["goal_pos"])

    # Taking the same action should result in the same next state
    action = 1
    next_obs1, _, _, _, _ = env1.step(action)
    next_obs2, _, _, _, _ = env2.step(action)
    np.testing.assert_allclose(next_obs1, next_obs2, atol=1e-5)


def test_different_personas_reward():
    """Tests that different personas yield different rewards."""
    # Speedrunner persona - testing goal completion
    env_speedrunner = OctogoneEnv(reward_persona="speedrunner", seed=42)
    env_speedrunner.reset()
    # Place player on the goal platform and ensure 'is_on_floor' is true
    goal_platform = env_speedrunner.level["platforms"][-1]["rect"]
    env_speedrunner.player_pos = np.array(
        [goal_platform[0] + goal_platform[2] / 2, goal_platform[1] + goal_platform[3]]
    )
    env_speedrunner.is_on_floor = True
    env_speedrunner.player_vel = np.array([0.0, 0.0])

    # Move towards the goal to trigger completion
    env_speedrunner.player_pos[0] = env_speedrunner.goal_pos[0] - 5
    _, reward_speedrunner, done_s, _, _ = env_speedrunner.step(1)  # move right
    assert done_s, "Speedrunner should finish the level"
    assert reward_speedrunner > 50, "Speedrunner should get a large completion bonus"

    # Survivor persona - testing death penalty
    env_survivor = OctogoneEnv(reward_persona="survivor", seed=42)
    env_survivor.reset()
    # Simulate falling off the world
    env_survivor.player_pos[1] = -300
    _, reward_survivor, done_v, _, _ = env_survivor.step(0)  # Action doesn't matter
    assert done_v, "Survivor should die from falling"
    assert reward_survivor < -40, "Survivor should get a large penalty for dying"

    assert reward_speedrunner != reward_survivor
