"""Universal Gymnasium wrapper with pluggable reward & HUD."""
from __future__ import annotations
import os, pygame, gymnasium as gym
import numpy as np
from typing import Callable, Any

Obs = Any
Info = dict

class GameEnv(gym.Env):
    metadata = {"render_modes": ["none", "human"]}

    def __init__(
        self,
        game_cls: type,               # subclass of games.*.FlappyCore-like
        *,
        render_mode: str = "none",
        fps: int | None = 30,
        max_steps: int | None = None,
        reward_fn: Callable[[Obs, float | None, bool, Info], float] | None = None,
        hud_fn: Callable[[pygame.Surface, "GameEnv"], None] | None = None,
        **game_kwargs,
    ):
        assert render_mode in self.metadata["render_modes"]
        self.game = game_cls(**game_kwargs)
        # self.game = game_cls  # not game_cls(**game_kwargs)
        self.render_mode = render_mode
        self.fps = fps
        self.max_steps = max_steps
        self.reward_fn = reward_fn or self._default_reward
        self.hud_fn = hud_fn

        # spaces from game
        self.action_space = self.game.get_action_space()
        self.observation_space = self.game.get_observation_space()

        # episode counters
        self._step_count = 0

        # GUI lazy data
        self.screen = None
        self.clock = None
        self.font = None

    # -------------------------------- default reward
    @staticmethod
    def _default_reward(obs, base, terminated, info):
        return 0.0 if base is None else base

    # -------------------------------- Gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        obs = self.game.reset()
        return obs, {}

    def step(self, action):
        self._step_count += 1

        # --- NEW: normalize action once, do NOT cast to bool unconditionally
        if hasattr(self.action_space, "n"):
            # Discrete
            if isinstance(action, (np.generic, np.ndarray)):  # e.g., numpy scalar from SB3
                action = int(action)
        # For Binary(1)/MultiBinary just pass through
        # For Box actions (not used here), pass as-is

        # OLD (remove): obs, base, terminated, info = self.game.step(bool(action))
        #print(f"[DEBUG] Using reward_fn: {self.reward_fn.__name__}")
        obs, base, terminated, info = self.game.step(action)

        truncated = bool(self.max_steps and self._step_count >= self.max_steps)
        reward = self.reward_fn(obs, base, terminated, info)
        #print(f"[DEBUG] step: base={base} persona={reward}")
        return obs, reward, terminated, truncated, info


    # -------------------------------- rendering
    def render(self):
        if self.render_mode != "human":
            return
        if self.screen is None:
            if os.environ.get("DISPLAY", "") == "":
                os.environ["SDL_VIDEODRIVER"] = "dummy"
            pygame.init()
            self.screen = pygame.display.set_mode((self.game.WIDTH, self.game.HEIGHT))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 20)

        # world
        self.game.render(self.screen, blit_only=True)

        # optional HUD from caller
        if self.hud_fn:
            self.hud_fn(self.screen, self)

        if self.fps and self.clock:
            self.clock.tick(self.fps)
        pygame.display.flip()

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None