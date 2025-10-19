# code/scripts/manual_play.py
import argparse
import importlib
import os
from pathlib import Path

import pygame

from code.wrappers.generic_env import GameEnv

os.environ["SDL_VIDEO_WINDOW_POS"] = "100,100"

parser = argparse.ArgumentParser()
parser.add_argument("--game", default="flappy", help="game key (e.g., flappy, tetris, asteroids)")
parser.add_argument("--fps", type=int, default=30, help="target FPS for manual play")
args = parser.parse_args()

# --- Load game core class dynamically (e.g., code.games.flappy_core.FlappyCore)
game_mod = importlib.import_module(f"code.games.{args.game}_core")
GameCls = getattr(game_mod, next(attr for attr in dir(game_mod) if attr.endswith("Core")))

# --- Init pygame BEFORE key handling
pygame.init()
clock = pygame.time.Clock()

# --- Global action state for event-based games
current_action = 0

# --- Helpful control text per game
def _handle_tetris_events(event) -> int:
    """
    Event-driven tetris controls - only triggers on KEYDOWN
    Returns action or -1 if no tetris action
    """
    if event.type != pygame.KEYDOWN:
        return -1
    
    if event.key == pygame.K_SPACE:
        return 5  # hard drop
    elif event.key in [pygame.K_UP, pygame.K_x, pygame.K_w]:
        return 3  # rotate
    elif event.key == pygame.K_LEFT:
        return 1  # move left
    elif event.key == pygame.K_RIGHT:
        return 2  # move right
    elif event.key in [pygame.K_DOWN, pygame.K_s]:
        return 4  # soft drop
    return -1  # no action

def _asteroids_action(keys: pygame.key.ScancodeWrapper) -> int:
    """
    Discrete action mapping for AsteroidsCore:
      0: noop
      1: turn left
      2: turn right
      3: thrust
      4: shoot
      5: hyperspace
    Priority: hyperspace > shoot > thrust > turn > noop
    """
    if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
        return 5  # hyperspace
    if keys[pygame.K_SPACE]:
        return 4  # shoot
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        return 3  # thrust
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        return 1  # turn left
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        return 2  # turn right
    return 0      # noop

ACTION_MAPPING = {
    # Flappy: 0 = idle, 1 = flap
    "flappy": lambda keys: 1 if keys[pygame.K_SPACE] else 0,
    # Asteroids (AsteroidsCore 6-action scheme)
    "asteroids": _asteroids_action,
}

CONTROL_DESCRIPTIONS = {
    "flappy": "SPACE = flap, ESC = quit",
    "tetris": "←/→ move, ↑/X/W rotate, ↓/S soft drop, SPACE hard drop, ESC quit (NO KEY REPEAT)",
    "asteroids": "←/→/A/D turn, ↑/W thrust, SPACE shoot, SHIFT hyperspace, ESC quit",
}

controls = CONTROL_DESCRIPTIONS.get(args.game, "Use game-specific keys. ESC = quit")
print(f"Use controls: {controls}")

# --- Build env; render_mode='human' makes a visible window; fps keeps time stable
env = GameEnv(GameCls, render_mode="human", fps=args.fps)

obs, _ = env.reset()
running = True
while running:
    # Keep a steady frame rate so physics are consistent
    clock.tick(args.fps)

    # Reset action to noop each frame
    action = 0

    # Poll events once per frame
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False
        
        # Handle tetris events (no key repeat)
        if args.game == "tetris":
            tetris_action = _handle_tetris_events(event)
            if tetris_action != -1:
                action = tetris_action

    # For non-tetris games, use continuous key polling
    if args.game != "tetris":
        keys = pygame.key.get_pressed()
        action_fn = ACTION_MAPPING.get(args.game, lambda k: 0)
        action = action_fn(keys)

    # Step the environment
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    # If the episode ends, auto-reset so you can keep playing until ESC
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
pygame.quit()
print("Game session ended.")
