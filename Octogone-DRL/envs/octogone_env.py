import gymnasium as gym
from gymnasium import spaces
import numpy as np

class OctogoneEnv(gym.Env):
    """
    A custom Gym environment for the game Octogone.
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, reward_persona='speedrunner', render_mode=None,
                 num_platforms=10, num_spikes=5, max_gap_size=200, seed=None):
        super(OctogoneEnv, self).__init__()

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None

        if self.render_mode == "human":
            import pygame
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((1280, 720))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

        # Level generation parameters
        self.num_platforms = num_platforms
        self.num_spikes = num_spikes
        self.max_gap_size = max_gap_size
        self.rng = np.random.default_rng(seed)

        # Define game constants with more realistic scaling
        self.max_speed = 30
        self.acceleration_air = 80
        self.acceleration_ground = 100
        self.jump_velocity = 20
        self.air_friction = 20
        self.ground_friction = 50
        self.gravity = 40
        self.max_fall = 100

        # Define action and observation space (adding last_momentum)
        self.action_space = spaces.Discrete(3)  # 0: left, 1: right, 2: jump
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32) # pos_x, pos_y, vel_x, vel_y, time, momentum

        # Environment state
        self.player_pos = np.array([0.0, 0.0])
        self.player_vel = np.array([0.0, 0.0])
        self.elapsed_time = 0.0
        self.is_on_floor = False
        self.reward_persona = reward_persona
        self.last_momentum = 0.0

        # Level data
        self.level = self._generate_level()
        self.goal_pos = self.level['goal_pos']
        self.platforms = self.level['platforms']
        self.spikes = self.level['spikes']

    def _generate_level(self):
        """Generates a random level with platforms, gaps, and spikes."""
        platforms = [{'rect': [-100, 0, 200, 50]}]
        spikes = []

        last_x = 100
        for _ in range(self.num_platforms):
            gap = self.rng.uniform(50, self.max_gap_size)
            width = self.rng.uniform(100, 300)
            height = self.rng.uniform(-100, 100)
            last_x += gap
            platforms.append({'rect': [last_x, height, width, 50]})
            last_x += width

        for _ in range(self.num_spikes):
            platform = self.rng.choice(platforms)
            spikes.append({'rect': [platform['rect'][0], platform['rect'][1] + 50, platform['rect'][2], 20]})

        goal_platform = platforms[-1]
        goal_pos = np.array([goal_platform['rect'][0] + goal_platform['rect'][2] / 2, goal_platform['rect'][1] + 50])

        return {'platforms': platforms, 'spikes': spikes, 'goal_pos': goal_pos}

    def step(self, action):
        """
        Execute one time step within the environment.
        """
        self.elapsed_time += 1.0 / 60.0  # Assuming 60 FPS

        # Apply actions
        direction = 0
        if action == 0:
            direction = -1
        elif action == 1:
            direction = 1

        if action == 2: # Jump
            if self.is_on_floor:
                self.player_vel[1] = self.jump_velocity

        # Apply physics
        if self.is_on_floor:
            self.player_vel[0] += direction * self.acceleration_ground * (1.0 / 60.0)
            self.player_vel[0] = np.clip(self.player_vel[0], -self.max_speed, self.max_speed)
            # Apply ground friction
            self.player_vel[0] *= (1.0 - self.ground_friction / 10000.0)
        else: # In air
            self.player_vel[0] += direction * self.acceleration_air * (1.0 / 60.0)
            self.player_vel[0] = np.clip(self.player_vel[0], -self.max_speed, self.max_speed)
            # Apply air friction
            self.player_vel[0] *= (1.0 - self.air_friction / 10000.0)
            # Apply gravity
            self.player_vel[1] -= self.gravity * (1.0 / 60.0)
            self.player_vel[1] = np.clip(self.player_vel[1], -self.max_fall, np.inf)

        # Update position
        self.player_pos += self.player_vel * (1.0 / 60.0)

        # Check for collisions
        self.is_on_floor = False
        done = False

        # Platforms
        for p in self.platforms:
            rect = p['rect']
            if (self.player_pos[0] + 10 > rect[0] and self.player_pos[0] - 10 < rect[0] + rect[2] and
                self.player_pos[1] + 20 > rect[1] and self.player_pos[1] < rect[1] + rect[3]):

                # Check side collisions
                if (self.player_pos[1] > rect[1] and self.player_pos[1] < rect[1] + rect[3]):
                    if self.player_vel[0] > 0:
                        self.player_pos[0] = rect[0] - 10
                    else:
                        self.player_pos[0] = rect[0] + rect[2] + 10
                    self.player_vel[0] = 0

                # Check top/bottom collisions
                elif self.player_vel[1] <= 0 and self.player_pos[1] > rect[1]: # Top collision
                    self.player_pos[1] = rect[1] + rect[3]
                    self.player_vel[1] = 0
                    self.is_on_floor = True
                elif self.player_vel[1] > 0: # Bottom collision
                    self.player_pos[1] = rect[1] - 20
                    self.player_vel[1] = 0

        # Spikes
        for s in self.spikes:
            rect = s['rect']
            if (self.player_pos[0] + 10 > rect[0] and self.player_pos[0] - 10 < rect[0] + rect[2] and
                self.player_pos[1] + 20 > rect[1] and self.player_pos[1] < rect[1] + rect[3]):
                done = True

        # Check for goal
        if np.linalg.norm(self.player_pos - self.goal_pos) < 20.0:
            done = True

        # Check for death by falling
        if self.player_pos[1] < -200:
            done = True

        # Update momentum
        self.last_momentum = np.linalg.norm(self.player_vel)

        # Calculate reward
        reward = self._calculate_reward(done)

        # Return state, reward, done, and info
        state = np.array([self.player_pos[0], self.player_pos[1], self.player_vel[0], self.player_vel[1], self.elapsed_time, self.last_momentum], dtype=np.float32)
        return state, reward, done, False, {}

    def _calculate_reward(self, done):
        """
        Calculate the reward based on the current persona.
        """
        reward = 0
        if self.reward_persona == 'speedrunner':
            reward -= 0.1  # Small penalty for each step to encourage speed
            if done:
                reward += 1000.0 - self.elapsed_time * 10
        elif self.reward_persona == 'survivor':
            reward += np.linalg.norm(self.player_vel) * 0.01 # Reward for movement
            if self.player_pos[1] < -100: # Penalty for falling off the level
                reward -= 100
        return reward

    def reset(self, seed=None, options=None):
        """
        Reset the state of the environment to an initial state.
        """
        super().reset(seed=seed)

        # Generate a new level
        self.level = self._generate_level()
        self.goal_pos = self.level['goal_pos']
        self.platforms = self.level['platforms']
        self.spikes = self.level['spikes']

        self.player_pos = np.array([0.0, 50.0]) # Start on the first platform
        self.player_vel = np.array([0.0, 0.0])
        self.elapsed_time = 0.0
        self.is_on_floor = True
        self.last_momentum = 0.0
        state = np.array([self.player_pos[0], self.player_pos[1], self.player_vel[0], self.player_vel[1], self.elapsed_time, self.last_momentum], dtype=np.float32)
        return state, {}

    def render(self):
        if self.render_mode == "human":
            import pygame
            self.screen.fill((0, 0, 0))

            # Draw platforms
            for p in self.platforms:
                rect = p['rect']
                pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(rect[0] - self.player_pos[0] + 640, rect[1] - self.player_pos[1] + 360, rect[2], rect[3]))

            # Draw spikes
            for s in self.spikes:
                rect = s['rect']
                pygame.draw.rect(self.screen, (255, 0, 0), pygame.Rect(rect[0] - self.player_pos[0] + 640, rect[1] - self.player_pos[1] + 360, rect[2], rect[3]))

            # Draw goal
            pygame.draw.circle(self.screen, (0, 255, 0), (self.goal_pos[0] - self.player_pos[0] + 640, self.goal_pos[1] - self.player_pos[1] + 360), 10)

            # Draw player
            pygame.draw.rect(self.screen, (0, 0, 255), pygame.Rect(630, 340, 20, 20))

            # Display info
            text = self.font.render(f"Time: {self.elapsed_time:.2f}", True, (255, 255, 255))
            self.screen.blit(text, (10, 10))

            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def print_level_layout(self):
        """Prints a simple ASCII representation of the level layout."""
        print("=" * 20)
        print("Level Layout:")
        for p in self.platforms:
            print(f"  Platform: {p['rect']}")
        for s in self.spikes:
            print(f"  Spike: {s['rect']}")
        print(f"  Goal: {self.goal_pos}")
        print("=" * 20)

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()

if __name__ == '__main__':
    env = OctogoneEnv(render_mode='human')
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        env.render()
    env.close()
