import subprocess 
import webbrowser
import os
import sys
import subprocess
import platform
from pathlib import Path
import shutil
import time
import yaml
import winsound


DEFAULT_TB_ROOT = "mylogs"   # match train.py tb_root
MODELS_DIR = Path("models/")
CONF_ROOT = Path("code/conf")
CONF_GAME_DIR = CONF_ROOT / "game"
CONF_REWARD_DIR = CONF_ROOT / "reward"
ALGO_CONF_DIR = CONF_ROOT / "algo"

# Set available algorithms (auto-detect from algo config folder if you want)
models_available = sorted([f.stem for f in ALGO_CONF_DIR.glob("*.yaml")])
current_model = models_available[0] if models_available else "ppo"
required_packages = [
    "numpy",
    "pygame",
    "matplotlib",
    "tqdm",
    "stable-baselines3",
    "gymnasium",
    # Add more as needed
]

def install_required(packages):
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

def get_skills_from_grid_yaml(path="code/conf/grid.yaml"):
    with open(path, "r") as f:
        grid = yaml.safe_load(f)
        skills = grid.get("skills", {})
    # Only keep the keys (skill names), not training steps.
    return list(skills.keys())

def get_available_games():
    if not CONF_GAME_DIR.exists():
        return []
    return sorted([f.stem for f in CONF_GAME_DIR.glob("*.yaml")])

def get_available_personas():
    if not CONF_REWARD_DIR.exists():
        return []
    return sorted([f.stem for f in CONF_REWARD_DIR.glob("*.yaml")])

def get_personas_for_game(game: str):
    all_personas = get_available_personas()
    filtered = [p for p in all_personas if p.startswith(f"{game}_")]
    return filtered if filtered else all_personas

def get_trained_games_from_models_flat():
    if not MODELS_DIR.exists():
        return []
    games = set()
    for z in MODELS_DIR.glob("*.zip"):
        parts = z.stem.split("_")
        if len(parts) >= 4:
            games.add(parts[0])
    return sorted(games)

def open_browser(url):
    try:
        webbrowser.open(url)
    except Exception:
        try:
            if platform.system() == "Linux":
                os.system(f"xdg-open {url}")
            elif platform.system() == "Windows":
                os.system(f"start {url}")
            elif platform.system() == "Darwin":
                os.system(f"open {url}")
        except Exception:
            pass
    print(f"If your browser did not open automatically, go to: {url}")

def ask_index(prompt, options, add_back=True):
    if not options:
        print("No options available.")
        return None
    print(prompt)
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    back_idx = len(options) + 1
    if add_back:
        print(f"  {back_idx}. Back")
    choice = input(f"Select (1-{back_idx if add_back else len(options)}): ").strip()
    try:
        num = int(choice)
        if add_back and num == back_idx:
            return None
        if 1 <= num <= len(options):
            return options[num - 1]
    except ValueError:
        pass
    print("Invalid selection.")
    return None

# ----- Actions -----

def run_training():
    global current_model
    print(f"\n=== Training (RL Algorithm: {current_model.upper()}) ===")
    games = get_available_games()
    if not games:
        print("No game configurations found in code/conf/game/")
        return
    game = ask_index("Available games:", games)
    if game is None:
        return
    personas = get_personas_for_game(game)
    if not personas:
        print(f"No personas found for game='{game}'. Create YAMLs like code/conf/reward/{game}_*.yaml")
        return
    persona_choice = ask_index("Available personas:", personas)
    if persona_choice is None:
        return
    skills = ["Novice", "Expert", "Custom steps"]
    skill_choice = ask_index("Skill / steps:", skills)
    if skill_choice is None:
        return
    tb_root = input(f"TensorBoard log root [{DEFAULT_TB_ROOT}]: ").strip() or DEFAULT_TB_ROOT
    game_arg    = f"game={game}"
    model_arg   = f"model={current_model}"
    persona_arg = f"persona={persona_choice}"
    if skill_choice == "Custom steps":
        steps_str = input("Enter total steps (e.g., 300000): ").strip()
        try:
            steps = int(steps_str)
        except ValueError:
            print("Invalid number.")
            return
        cmd = [
            sys.executable, "-m", "code.scripts.train",
            game_arg, model_arg, persona_arg,
            "skill=Custom",
            f"+skills.Custom={steps}",
            f"tb_root={tb_root}",
        ]
    else:
        skill_arg = f"skill={skill_choice}"
        cmd = [
            sys.executable, "-m", "code.scripts.train",
            game_arg, model_arg, persona_arg, skill_arg,
            f"tb_root={tb_root}",
        ]
    print("\n>>>", " ".join(cmd), "\n")
    subprocess.run(cmd)
    winsound.PlaySound("chime.wav", winsound.SND_FILENAME)
    print("\nTraining completed.\n")

def run_evaluation():
    print("\n===== Evaluation =====")
    BEST_DIR = MODELS_DIR / "best"
    if not BEST_DIR.exists():
        print("[!] models/best/ does not exist — please train some models first.")
        return
    subfolders = [p for p in BEST_DIR.iterdir() if p.is_dir() and (p / "best_model.zip").exists()]
    if not subfolders:
        print("No best_model.zip files found in models/best/.")
        return
    games = sorted(set(f.name.split("_")[0] for f in subfolders))
    if not games:
        print("No recognized games found in models/best/.")
        return
    game = ask_index("Games with trained models", games)
    if game is None:
        return
    model_dirs = [f for f in subfolders if f.name.startswith(game)]
    if not model_dirs:
        print(f"No best_model.zip folders found for game='{game}' in {BEST_DIR}")
        return
    print(f"\nFound {len(model_dirs)} model(s) for '{game}'. Running quick eval (25 eps each)...\n")
    for model_dir in model_dirs:
        model_zip = model_dir / "best_model.zip"
        model_name = model_dir.name
        parts = model_name.split("_")
        algo = parts[1] if len(parts) > 1 else current_model
        out_json = MODELS_DIR / f"{model_name}_eval.json"
        metrics_class = f"code.metrics.{game}_balance.{game.capitalize()}BalanceStats"
        cmd = [
            sys.executable, "-m", "code.scripts.evaluate",
            "--game", game,
            "--algo", algo,
            "--model", str(model_zip),
            "--episodes", "100",
            "--render", "none",
            "--out", str(out_json),
            "--metrics", metrics_class,
        ]
        print(">>>", " ".join(cmd))
        subprocess.run(cmd)
    print("\n✓ Evaluation completed for all best models.\n")

def train_all_models():
    global current_model
    print(f"\n===== Train ALL Models for Selected Game (RL Algorithm: {current_model.upper()}) =====")
    games = get_available_games()
    if not games:
        print("No game configurations found in code/conf/game/")
        return
    game = ask_index("Available games:", games)
    if game is None:
        return
    personas = get_personas_for_game(game)
    if not personas:
        print(f"[skip] No personas for game '{game}' (expect {game}_*.yaml)")
        return
    skills = get_skills_from_grid_yaml()
    if not skills:
        print(f"[skip] No skills defined in grid.yaml")
        return
    tb_root = DEFAULT_TB_ROOT
    total_runs = 0
    for persona in personas:
        for skill in skills:
            cmd = [
                sys.executable, "-m", "code.scripts.train",
                f"game={game}",
                f"model={current_model}",
                f"persona={persona}",
                f"skill={skill}",
                f"tb_root={tb_root}",
            ]
            print(f"\n>>> Training: {game} | Persona: {persona} | Skill: {skill} | Algo: {current_model.upper()}")
            print(" ".join(cmd))
            subprocess.run(cmd)
            total_runs += 1
    winsound.PlaySound("chime.wav", winsound.SND_FILENAME)
    print(f"\nCompleted training for {total_runs} model(s) in game '{game}'.\n")

def watch_trained_agent():
    BEST_DIR = MODELS_DIR / "best"
    if not BEST_DIR.exists():
        print("[!] models/best/ does not exist — please train some models first.")
        return
    model_folders = [f for f in BEST_DIR.iterdir() if f.is_dir() and (f / "best_model.zip").exists()]
    if not model_folders:
        print("No best_model.zip files found in models/best/.")
        return
    display_options = []
    paths = []
    for folder in model_folders:
        parts = folder.name.split("_")
        if len(parts) >= 5:
            game, algo, _g2, persona, skill = parts[0], parts[1], parts[2], parts[3], parts[4]
            display = f"{game:<8} | {algo:<4} | {persona:<12} | {skill:<8}"
        else:
            display = folder.name
        display_options.append(display)
        paths.append(folder / "best_model.zip")
    idx = ask_index("Select a trained model to visualize (AI agent will play with rendering):", display_options)
    if idx is None:
        return
    model_path = paths[display_options.index(idx)] if idx in display_options else paths[int(idx)-1]
    episodes = input("How many episodes to watch? [10]: ").strip()
    episodes = int(episodes) if episodes.isdigit() and int(episodes) > 0 else 10
    fps = input("Rendering FPS? [30]: ").strip()
    fps = int(fps) if fps.isdigit() and int(fps) > 0 else 30
    cmd = [
        sys.executable, "-m", "code.scripts.watch_agent",
        str(model_path), 
        "--episodes", str(episodes),
        "--fps", str(fps)
    ]
    print("\n>>>", " ".join(str(x) for x in cmd), "\n")
    subprocess.run(cmd)
    print("\nVisualization completed.\n")

def run_tensorboard():
    print("\n=== TensorBoard (auto-open, blocking) ===")
    root = Path(DEFAULT_TB_ROOT)
    if not root.exists():
        print(f"No '{DEFAULT_TB_ROOT}/' folder found yet. Train first or change tb_root in train.")
        return
    games = get_available_games()
    filter_game = ask_index("Optional: choose a game to filter (or Back to show all):", games)
    for port in range(6006, 6011):
        cmd = [
            sys.executable, "-m", "tensorboard.main",
            "--logdir", str(root),
            "--port", str(port),
        ]
        print("\nLaunching TensorBoard…")
        print(">>>", " ".join(cmd))
        print(f"\nTensorBoard will run on: http://localhost:{port}/")
        print("Opening browser in a few seconds… (Press Ctrl+C here to stop TensorBoard)\n")
        try:
            tb_proc = subprocess.Popen(cmd)
            time.sleep(3)
            open_browser(f"http://localhost:{port}/")
            if filter_game:
                print(f"\nTip: In TensorBoard’s left panel, use the run filter: .*{filter_game}.*")
            tb_proc.wait()
            return
        except KeyboardInterrupt:
            print("\nTensorBoard stopped by user.")
            return
        except FileNotFoundError:
            print("TensorBoard not found. Install with: pip install tensorboard")
            return
        except Exception as e:
            print(f"Error launching TB on port {port}: {e}")
    print("Unable to start TensorBoard on ports 6006–6010.")

def delete_logs_and_models():
    print("\n=== Delete TensorBoard Logs and Models ===")
    confirm = input(
        f"This will permanently delete '{DEFAULT_TB_ROOT}/' and '{MODELS_DIR}/'.\n"
        "Are you sure you want to continue? [y/N]: "
    ).strip().lower()
    if confirm not in ("y", "yes"):
        print("Aborted. Nothing was deleted.")
        return
    def safe_clear_dir(path: Path):
        if not path.exists():
            return
        for attempt in range(3):
            try:
                shutil.rmtree(path)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"Failed to delete {path}: {e}")
                    return
                print(f"{path} might be in use. Retrying in 1s…")
                time.sleep(1)
        try:
            path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Cleared and recreated {path}")
        except Exception as e:
            print(f"Deleted {path} but failed to recreate it: {e}")
    safe_clear_dir(Path(DEFAULT_TB_ROOT))
    safe_clear_dir(MODELS_DIR)
    print("\n🧹 All logs and models deleted successfully.\n")

def run_manual_play():
    available_games = get_available_games()
    if not available_games:
        print("No game configurations found in code/conf/game/")
        return
    print("Available games:")
    for i, game in enumerate(available_games, 1):
        print(f"  {i}. Play {game}")
    print(f"  {len(available_games) + 1}. Back to main menu")
    choice = input(f"Select game to play (1-{len(available_games) + 1}): ").strip()
    try:
        idx = int(choice)
    except ValueError:
        print("Invalid selection.")
        return
    if idx == len(available_games) + 1:
        return
    if not (1 <= idx <= len(available_games)):
        print("Invalid selection.")
        return
    selected_game = available_games[idx - 1]
    print(f"\n=== Playing {selected_game} manually ===")
    print("Use game controls (e.g., spacebar for Flappy Bird). Press ESC to quit.")
    script_path = Path("code/scripts/manual_play.py")
    if not script_path.exists():
        print("Manual play script not found at code/scripts/manual_play.py")
        return
    subprocess.run([sys.executable, "-m", "code.scripts.manual_play", "--game", selected_game, "--fps", "30"])

def show_project_status():
    games = get_available_games()
    trained = get_trained_games_from_models_flat()
    print(f"Available game configurations: {len(games)}")
    for g in games:
        flag = "✓ Trained" if g in trained else "○ Not trained"
        print(f"   {g}: {flag}")
    if MODELS_DIR.exists():
        zips = list(MODELS_DIR.glob("*.zip"))
        print(f"\nTotal trained models: {len(zips)}")
    else:
        print("\n✗ No models directory found")
    if CONF_ROOT.exists():
        print("\nConfiguration status:")
        for sub in ["algo", "game", "reward"]:
            p = CONF_ROOT / sub
            n = len(list(p.glob("*.yaml"))) if p.exists() else 0
            print(f"   {sub}: {n} configs")
    else:
        print("\n✗ Configuration directory not found")
    print()

# ---------- Menu ----------

def main():
    install_required(required_packages)
    global current_model
    while True:
        print("=" * 60)
        print(f"MULTI-GAME RL TRAINING & EVALUATION MENU   (RL Algorithm: {current_model.upper()})")
        print("=" * 60)
        games = get_available_games()
        trained = get_trained_games_from_models_flat()
        print(f"Games available: {len(games)} | Games trained: {len(trained)}")
        print("\nOptions:")
        print("1. Run Training (pick Game, Persona, Skill)")
        print("2. Run Evaluation (per-game, scans models/*.zip)")
        print("3. View TensorBoard Logs (mylogs/)")
        print("4. Play Game Manually (keyboard)")
        print("5. Show Detailed Project Status")
        print("6. Watch Trained Agent Play (visualize AI performance)")
        print("7. Train All Models (all combos for one game)")
        print("8. Delete TensorBoard Logs & Models")
        print("9. Switch RL Algorithm (current: {})".format(current_model.upper()))
        print("10. Exit")
        choice = input("Select option (1-10): ").strip()
        if choice == "1":
            run_training()
        elif choice == "2":
            run_evaluation()
        elif choice == "3":
            run_tensorboard()
        elif choice == "4":
            run_manual_play()
        elif choice == "5":
            show_project_status()
        elif choice == "6":
            watch_trained_agent()
        elif choice == "7":
            train_all_models()
        elif choice == "8":
            delete_logs_and_models()
        elif choice == "9":
            new_model = ask_index("Select RL Algorithm:", models_available, add_back=True)
            if new_model:
                current_model = new_model
        elif choice == "10":
            print("Exiting. Happy training!")
            break
        else:
            print("Invalid selection. Please choose 1-10.\n")

if __name__ == "__main__":
    main()
