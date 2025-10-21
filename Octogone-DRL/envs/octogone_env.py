import gymnasium as gym
from gymnasium import spaces
import numpy as np


class OctogoneEnv(gym.Env):
    """
    A custom Gym environment for the game Octogone with procedural level generation.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        reward_persona="speedrunner",
        render_mode=None,
        difficulty=0,
        seed=None,
        **kwargs,
    ):
        super(OctogoneEnv, self).__init__()

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        self.difficulty = difficulty

        if self.render_mode == "human":
            import pygame

            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((1280, 720))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

        self.rng = np.random.default_rng(seed)

        # Allow overriding level gen params
        self.num_platforms = kwargs.get("num_platforms", 5 + self.difficulty * 2)
        self.num_spikes = kwargs.get("num_spikes", 2 + self.difficulty)
        self.max_gap_x = kwargs.get("max_gap_size", 100 + self.difficulty * 20)

        # Define game constants with more realistic scaling
        self.max_speed = 30
        self.acceleration_air = 80
        self.acceleration_ground = 100
        self.jump_velocity = 20
        self.air_friction = 20
        self.ground_friction = 50
        self.gravity = 40
        self.max_fall = 100

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        self.reward_persona = reward_persona
        self.reset()

    def set_difficulty(self, difficulty):
        self.difficulty = difficulty

    def _generate_level(self):
        max_gap_y = 50 + self.difficulty * 10

        platforms = [{"rect": [-200, 0, 400, 50]}]
        spikes = []

        last_platform = platforms[0]
        for _ in range(self.num_platforms):
            px, py, pw, ph = last_platform["rect"]
            nx = px + pw + self.rng.uniform(50, self.max_gap_x)
            ny = self.rng.uniform(max(-200, py - max_gap_y), min(200, py + max_gap_y))
            nw = self.rng.uniform(100, 300)
            new_platform = {"rect": [nx, ny, nw, 50]}
            platforms.append(new_platform)
            last_platform = new_platform

        for _ in range(self.num_spikes):
            platform = self.rng.choice(platforms[1:])
            sx, sy, sw, sh = platform["rect"]
            spikes.append({"rect": [sx, sy + sh, sw, 20]})

        goal_platform = platforms[-1]
        goal_pos = np.array(
            [
                goal_platform["rect"][0] + goal_platform["rect"][2] / 2,
                goal_platform["rect"][1] + 50,
            ]
        )

        start_platform = self.rng.choice(platforms[: len(platforms) // 2])
        start_pos = np.array(
            [
                start_platform["rect"][0] + start_platform["rect"][2] / 2,
                start_platform["rect"][1] + 50,
            ]
        )

        return {
            "platforms": platforms,
            "spikes": spikes,
            "goal_pos": goal_pos,
            "start_pos": start_pos,
        }

    def step(self, action):
        self.elapsed_time += 1.0 / 60.0

        current_pos_tuple = tuple(np.round(self.player_pos).astype(int))
        new_tile_visited = current_pos_tuple not in self.visited_tiles
        if new_tile_visited:
            self.visited_tiles.add(current_pos_tuple)

        direction = 0
        if action == 0:
            direction = -1
        elif action == 1:
            direction = 1

        if action == 2 and self.is_on_floor:
            self.player_vel[1] = self.jump_velocity

        if self.is_on_floor:
            self.player_vel[0] += direction * self.acceleration_ground * (1.0 / 60.0)
        else:
            self.player_vel[0] += direction * self.acceleration_air * (1.0 / 60.0)
        self.player_vel[0] = np.clip(
            self.player_vel[0], -self.max_speed, self.max_speed
        )

        if self.is_on_floor:
            self.player_vel[0] *= 1.0 - self.ground_friction / 10000.0
        else:
            self.player_vel[0] *= 1.0 - self.air_friction / 10000.0
            self.player_vel[1] -= self.gravity * (1.0 / 60.0)
        self.player_vel[1] = np.clip(self.player_vel[1], -self.max_fall, np.inf)

        # Separate physics updates for horizontal and vertical movement
        # Horizontal movement and collision
        self.player_pos[0] += self.player_vel[0] * (1.0 / 60.0)
        for p in self.platforms:
            rect = p["rect"]
            player_rect = [self.player_pos[0] - 10, self.player_pos[1], 20, 20]
            if (
                player_rect[0] < rect[0] + rect[2]
                and player_rect[0] + player_rect[2] > rect[0]
                and player_rect[1] < rect[1] + rect[3]
                and player_rect[1] + player_rect[3] > rect[1]
            ):
                if self.player_vel[0] > 0:  # moving right
                    self.player_pos[0] = rect[0] - 10.1
                elif self.player_vel[0] < 0:  # moving left
                    self.player_pos[0] = rect[0] + rect[2] + 10.1
                self.player_vel[0] = 0

        # Vertical movement and collision
        self.player_pos[1] += self.player_vel[1] * (1.0 / 60.0)
        self.is_on_floor = False
        for p in self.platforms:
            rect = p["rect"]
            player_rect = [self.player_pos[0] - 10, self.player_pos[1], 20, 20]
            if (
                player_rect[0] < rect[0] + rect[2]
                and player_rect[0] + player_rect[2] > rect[0]
                and player_rect[1] < rect[1] + rect[3]
                and player_rect[1] + player_rect[3] > rect[1]
            ):
                if self.player_vel[1] <= 0:  # moving down/on floor
                    self.player_pos[1] = rect[1] + rect[3]
                    self.player_vel[1] = 0
                    self.is_on_floor = True
                elif self.player_vel[1] > 0:  # moving up (hitting ceiling)
                    self.player_pos[1] = rect[1] - 20.1
                    self.player_vel[1] = 0

        done = False
        for s in self.spikes:
            rect = s["rect"]
            if (
                self.player_pos[0] + 10 > rect[0]
                and self.player_pos[0] - 10 < rect[0] + rect[2]
                and self.player_pos[1] + 20 > rect[1]
                and self.player_pos[1] < rect[1] + rect[3]
            ):
                done = True

        if np.linalg.norm(self.player_pos - self.goal_pos) < 20.0:
            done = True
        if self.player_pos[1] < -200:
            done = True

        self.last_momentum = np.linalg.norm(self.player_vel)
        reward = self._calculate_reward(done, new_tile_visited)
        state = np.concatenate(
            [self.player_pos, self.player_vel, [self.elapsed_time, self.last_momentum]]
        )
        return state.astype(np.float32), reward, done, False, {}

    def _calculate_reward(self, done, new_tile_visited):
        reward = 0
        if self.reward_persona == "speedrunner":
            reward -= 0.01  # Small penalty for time
            if new_tile_visited:
                reward += 0.02  # Small reward for exploration
            if done:
                if np.linalg.norm(self.player_pos - self.goal_pos) < 20.0:
                    # Large reward for reaching the goal, scaled by time
                    reward += 100 - self.elapsed_time * 2
                else:
                    # Large penalty for dying
                    reward -= 50
        elif self.reward_persona == "survivor":
            reward += 0.1  # Reward for surviving another step
            if new_tile_visited:
                reward += 0.05  # Small reward for exploration
            if done:
                if np.linalg.norm(self.player_pos - self.goal_pos) < 20.0:
                    # Survivor doesn't care much for the goal
                    reward += 10
                else:
                    # Large penalty for dying
                    reward -= 100
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.level = self._generate_level()
        self.goal_pos = self.level["goal_pos"]
        self.platforms = self.level["platforms"]
        self.spikes = self.level["spikes"]
        self.player_pos = self.level["start_pos"]
        self.player_vel = np.array([0.0, 0.0])
        self.elapsed_time = 0.0
        self.is_on_floor = True
        self.last_momentum = 0.0
        self.visited_tiles = set()
        state = np.concatenate(
            [self.player_pos, self.player_vel, [self.elapsed_time, self.last_momentum]]
        )
        return state.astype(np.float32), {}

    def render(self):
        if self.render_mode == "human":
            import pygame

            self.screen.fill((10, 10, 10))
            for p in self.platforms:
                pygame.draw.rect(
                    self.screen,
                    (200, 200, 200),
                    pygame.Rect(
                        p["rect"][0] - self.player_pos[0] + 640,
                        p["rect"][1] - self.player_pos[1] + 360,
                        p["rect"][2],
                        p["rect"][3],
                    ),
                )
            for s in self.spikes:
                pygame.draw.rect(
                    self.screen,
                    (255, 50, 50),
                    pygame.Rect(
                        s["rect"][0] - self.player_pos[0] + 640,
                        s["rect"][1] - self.player_pos[1] + 360,
                        s["rect"][2],
                        s["rect"][3],
                    ),
                )
            pygame.draw.circle(
                self.screen,
                (50, 255, 50),
                (
                    self.goal_pos[0] - self.player_pos[0] + 640,
                    self.goal_pos[1] - self.player_pos[1] + 360,
                ),
                15,
            )
            pygame.draw.rect(self.screen, (50, 50, 255), pygame.Rect(630, 340, 20, 20))
            text = self.font.render(
                f"Time: {self.elapsed_time:.2f}", True, (255, 255, 255)
            )
            self.screen.blit(text, (10, 10))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def print_level_layout(self):
        print("=" * 20 + "\nLevel Layout:\n" + "=" * 20)
        for i, p in enumerate(self.platforms):
            print(f"Platform {i}: {p['rect']}")
        for i, s in enumerate(self.spikes):
            print(f"Spike {i}: {s['rect']}")
        print(f"Start: {self.level['start_pos']}, Goal: {self.goal_pos}\n" + "=" * 20)

    def close(self):
        if self.screen:
            import pygame

            pygame.display.quit()
            pygame.quit()
