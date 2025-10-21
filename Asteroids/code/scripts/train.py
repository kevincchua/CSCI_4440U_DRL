from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3 import PPO, A2C
import os 
import sys 
import importlib
import inspect
from pathlib import Path
from typing import Any, Dict
import torch
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from code.wrappers.generic_env import GameEnv
from code.algos import get_algo
from stable_baselines3.common.vec_env import SubprocVecEnv

class AnnealCallback(BaseCallback):
    def __init__(self, total_timesteps, 
                start_ent=0.15, end_ent=0.01, 
                start_grad_clip=1.0, end_grad_clip=0.3, 
                start_lr=None, end_lr=None,
                verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.start_ent = start_ent
        self.end_ent = end_ent
        self.start_grad_clip = start_grad_clip
        self.end_grad_clip = end_grad_clip
        self.start_lr = start_lr
        self.end_lr = end_lr

    def _on_step(self) -> bool:
        frac = self.num_timesteps / self.total_timesteps
        # Entropy coefficient annealing
        ent_coef = self.start_ent * (1 - frac) + self.end_ent * frac
        self.model.ent_coef = ent_coef
        
        # Gradient clipping annealing
        max_grad_norm = self.start_grad_clip * (1 - frac) + self.end_grad_clip * frac
        self.model.max_grad_norm = max_grad_norm

        # Learning rate annealing (if chosen)
        if self.start_lr is not None and self.end_lr is not None:
            lr = self.start_lr * (1 - frac) + self.end_lr * frac
            # Update lr for different model APIs
            if hasattr(self.model, "lr_schedule"):
                self.model.lr_schedule = lambda _: lr
                # Also update optimizer param groups directly if needed
                if hasattr(self.model, "policy") and hasattr(self.model.policy, "optimizer"):
                    for param_group in self.model.policy.optimizer.param_groups:
                        param_group["lr"] = lr
            elif hasattr(self.model, "policy") and hasattr(self.model.policy, "optimizer"):
                for param_group in self.model.policy.optimizer.param_groups:
                    param_group["lr"] = lr

        return True


def _pretty_steps(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    return f"{n // 1_000}k"


def _load_yaml(conf_root: Path, group: str, name: str) -> Dict:
    """Load a YAML file like conf/<group>/<name>.yaml into a dict."""
    path = conf_root / group / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # dict


def _import_attr(dotted: str) -> Any:
    """Import and return the attribute given a dotted path 'pkg.mod.Attr'."""
    mod_path, attr = dotted.rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    return getattr(mod, attr)


def _resolve_callable_or_instance(node: Dict[str, Any]) -> Any:
    """
    Resolve a Hydra node for rewards/callbacks:
    {'_target_': 'pkg.mod.Attr', ...kwargs}
    - If Attr is a class, instantiate with kwargs.
    - If Attr is a function/callable, return it as-is (kwargs ignored).
    """
    if not isinstance(node, dict) or "_target_" not in node:
        raise ValueError(f"Bad hydra target node: {node}")
    obj = _import_attr(node["_target_"])
    if inspect.isclass(obj):
        kwargs = {k: v for k, v in node.items() if k != "_target_"}
        return obj(**kwargs)
    return obj
@hydra.main(version_base=None, config_path="../conf", config_name="grid")
def main(cfg: DictConfig):
    """
    Trains across models × personas × skills for a game.

    Usage:
        python -m code.scripts.train game=flappy
        # or edit code/conf/grid.yaml and run without overrides
    """
    # Stable paths regardless of Hydra's run dir
    repo_root = Path(get_original_cwd())
    conf_root = repo_root / "code" / "conf"
    models_dir = repo_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA is not available on this system, falling back to CPU.")
        device = "cpu"
    
    print(f"[INFO] Training device: {device}")

    # --- NEW: Logging/callback knobs (override-able from Hydra) ---
    # WHY: Making these configurable lets you run ablations and keep runs comparable.
    tb_root  = str(cfg.get("tb_root", "runs"))          # top-level TensorBoard folder
    eval_freq = int(cfg.get("eval_freq", 20_000))       # how often to run EvalCallback
    save_freq = int(cfg.get("save_freq", 50_000))       # how often to snapshot checkpoints

    # --- Build game CLASS (not instance)
    if isinstance(cfg.game, (DictConfig, dict)):
        game_conf = OmegaConf.to_container(cfg.game, resolve=True)
        if "_target_" in game_conf:
            dotted = game_conf["_target_"]  # e.g. "code.games.flappy_core.FlappyCore"
            game_name = dotted.split(".")[-2].replace("_core", "")  # → "flappy"
        else:
            game_name = "game"
    else:
        game_conf = _load_yaml(conf_root, "game", str(cfg.game))
        game_name = str(cfg.game)  # use the actual string passed in (e.g. "flappy")

    if "_target_" not in game_conf:
        raise ValueError(f"Game config must have _target_: {game_conf}")
    game_cls = _import_attr(game_conf["_target_"])  # <-- class, no instantiation

    # Shared env params (reward set per persona)
    base_env_kwargs = dict(
        render_mode=cfg.render_mode,
        fps=None if str(cfg.fps).lower() == "none" else int(cfg.fps),
        max_steps=None if str(cfg.max_steps).lower() == "none" else int(cfg.max_steps),
    )

    # Ensure deterministic folders for artifacts
    os.makedirs(models_dir / "best", exist_ok=True)         # WHY: EvalCallback best_model.zip
    os.makedirs(models_dir / "checkpoints", exist_ok=True)  # WHY: periodic checkpoints
    os.makedirs(models_dir / "eval_logs", exist_ok=True)    # WHY: eval metrics (CSV/JSON)

    # --- Optional single-value shortcuts (CLI-friendly) ---
    # You can pass: model=ppo persona=flappy_speedrunner skill=Expert
    selected_models = list(cfg.models)
    if "model" in cfg and cfg.model:
        selected_models = [str(cfg.model)]

    selected_personas = list(cfg.personas)
    if "persona" in cfg and cfg.persona:
        selected_personas = [str(cfg.persona)]

    selected_skills = dict(cfg.skills)
    if "skill" in cfg and cfg.skill:
        key = str(cfg.skill)
        if key not in selected_skills:
            raise ValueError(f"skill='{key}' not in cfg.skills {list(cfg.skills.keys())}")
        selected_skills = {key: selected_skills[key]}


    run_count = 0
    for model_name in selected_models:
        # Algo params from conf/algo/<model>.yaml
        algo_conf = _load_yaml(conf_root, "algo", model_name)
        Algo = get_algo(algo_conf.get("name", model_name))
        policy = algo_conf.get("policy", "MlpPolicy")
        algo_kwargs = {k: v for k, v in algo_conf.items() if k not in {"_target_", "name", "policy"}}

        for persona in selected_personas:
            # Reward function from conf/reward/<persona>.yaml
            reward_conf = _load_yaml(conf_root, "reward", persona)
            reward_fn = _resolve_callable_or_instance(reward_conf)

            # One env per persona; GameEnv will instantiate game_cls internally
            #env = GameEnv(game_cls, reward_fn=reward_fn, **base_env_kwargs)

            def make_env():
                def _init():
                    return GameEnv(game_cls, reward_fn=reward_fn, **base_env_kwargs)
                return _init

            n_envs = int(cfg.get("n_envs", 1))
            if n_envs > 1:
                env = SubprocVecEnv([make_env() for _ in range(n_envs)])
            else:
                env = GameEnv(game_cls, reward_fn=reward_fn, **base_env_kwargs)
            
            # --- NEW: dedicated eval env (no training noise) ---
            eval_env = GameEnv(game_cls, reward_fn=reward_fn, **base_env_kwargs)

            for skill, total_timesteps in selected_skills.items():
                run_count += 1


                # --- NEW: TB directory for this (game × algo × persona) ---
                # WHY: Separate folders make charts easy to compare in TensorBoard.
                tb_dir = os.path.join(tb_root, f"{game_name}_{model_name}_{persona}")
                os.makedirs(tb_dir, exist_ok=True)
                
                # --- NEW: EvalCallback for model evaluation ---
                
                eval_cb = EvalCallback(
                    eval_env,
                    best_model_save_path=str(models_dir / "best" / f"{game_name}_{model_name}_{persona}_{str(skill).lower()}"),
                    log_path=str(models_dir / "eval_logs" / f"{game_name}_{model_name}_{persona}_{str(skill).lower()}"),
                    eval_freq=eval_freq,
                    deterministic=True,
                    render=False,
                )
                
                # --- NEW: CheckpointCallback for model saving ---
                ckpt_cb = CheckpointCallback(
                    save_freq=save_freq,                              # WHY: safety net; resume mid-run if needed
                    save_path=str(models_dir / "checkpoints"),
                    name_prefix=f"{game_name}_{model_name}_{persona}"
                )
                anneal_configs = {
                    "ppo": dict(
                        start_ent=0.15, end_ent=0.0005,
                        start_grad_clip=0.5, end_grad_clip=0.3,
                        start_lr=0.0004, end_lr=0.00003
                    ),
                    "a2c": dict(
                        start_ent=0.15, end_ent=0.008,
                        start_grad_clip=0.5, end_grad_clip=0.5,
                        start_lr=0.0007, end_lr=0.00007
                    ),
                }

                policy_key = model_name.lower()
                config = anneal_configs.get(policy_key)
                if config is not None:
                    anneal_callback = AnnealCallback(total_timesteps=total_timesteps, **config)
                else:
                    anneal_callback = None

                # --- NEW: inject TensorBoard path into algo kwargs ---
                # WHY: SB3 reads this to write scalars; kept inside loop to isolate runs.
                train_kwargs = dict(algo_kwargs)
                train_kwargs["tensorboard_log"] = tb_dir
                train_kwargs["device"] = device  # <-- add this before model creation
                # Build model (SB3-compatible Algo expected)
                model = Algo(policy, env, **train_kwargs)

                # Label inside TB so runs are grouped by persona/skill
                tb_run_name = f"{model_name}_{persona}_{str(skill).lower()}"

                # Learn with callbacks + TB run label
                model.learn(
                    total_timesteps=int(total_timesteps),
                    callback=[eval_cb, ckpt_cb, anneal_callback],
                    tb_log_name=tb_run_name,  # WHY: neat grouping in TB UI
                    progress_bar=True
                )


            # Save each trained variant (SB3 appends .zip)
                filename = f"{game_name}_{model_name}_{persona}_{str(skill).lower()}.zip"
                save_path = models_dir / filename
                #model.save(str(save_path))
                
                print(f"[{run_count}] saved → {save_path}  ({_pretty_steps(int(total_timesteps))} steps)")
            
            

            # Close envs between personas
            try:
                env.close()
                eval_env.close()
            except Exception:
                pass

    print(f"Done. Trained {run_count} models for game='{game_name}'. Models at: {models_dir}")


if __name__ == "__main__":
    main()