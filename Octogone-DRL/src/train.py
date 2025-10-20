import yaml
import os
import sys
sys.path.append(os.getcwd())
from stable_baselines3 import PPO, A2C
from envs.octogone_env import OctogoneEnv
import gymnasium as gym

def train(config_file):
    """
    Train a model based on the given configuration file.
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Create the environment
    env = OctogoneEnv(reward_persona=config['env']['reward_persona'])

    # Create the model
    if config['model']['name'] == 'PPO':
        model = PPO(
            config['model']['policy'],
            env,
            learning_rate=config['model']['learning_rate'],
            n_steps=config['model']['n_steps'],
            batch_size=config['model']['batch_size'],
            n_epochs=config['model']['n_epochs'],
            gamma=config['model']['gamma'],
            gae_lambda=config['model']['gae_lambda'],
            clip_range=config['model']['clip_range'],
            ent_coef=config['model']['ent_coef'],
            verbose=config['model']['verbose'],
            seed=config['model']['seed'],
            tensorboard_log="./logs/ppo_tensorboard/"
        )
    elif config['model']['name'] == 'A2C':
        model = A2C(
            config['model']['policy'],
            env,
            learning_rate=config['model']['learning_rate'],
            n_steps=config['model']['n_steps'],
            gamma=config['model']['gamma'],
            gae_lambda=config['model']['gae_lambda'],
            ent_coef=config['model']['ent_coef'],
            vf_coef=config['model']['vf_coef'],
            max_grad_norm=config['model']['max_grad_norm'],
            use_rms_prop=config['model']['use_rms_prop'],
            verbose=config['model']['verbose'],
            seed=config['model']['seed'],
            tensorboard_log="./logs/a2c_tensorboard/"
        )
    else:
        raise ValueError(f"Unknown model: {config['model']['name']}")

    # Train the model
    model.learn(total_timesteps=config['model']['total_timesteps'])

    # Save the model
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    model_name = os.path.splitext(os.path.basename(config_file))[0]
    model.save(f"models/{model_name}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()
    train(args.config)
