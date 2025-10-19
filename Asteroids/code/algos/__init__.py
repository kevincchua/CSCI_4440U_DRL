"""Very thin registry so `hydra` configs can request an algo by name."""
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3, DDPG

ALGO_REGISTRY = {
    "ppo": PPO,
    "a2c": A2C,
    "dqn": DQN,
    "sac": SAC,
    "td3": TD3,
    "ddpg": DDPG,
}


def get_algo(name: str):
    try:
        return ALGO_REGISTRY[name.lower()]
    except KeyError as e:
        raise ValueError(f"Unknown algo '{name}'. Available: {list(ALGO_REGISTRY)}") from e