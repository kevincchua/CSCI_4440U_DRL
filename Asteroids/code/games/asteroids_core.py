from __future__ import annotations
import os
import math
import random
from dataclasses import dataclass
from typing import List
import numpy as np
import pygame
from gymnasium import spaces

# Game constants
SCREEN_SCALE = 1
WIDTH_CELLS, HEIGHT_CELLS = 800, 600
SCREEN_WIDTH = WIDTH_CELLS 
SCREEN_HEIGHT = HEIGHT_CELLS
UP = np.array([0, -1], dtype=np.float32)
MAX_ASTEROIDS = 50
MAX_BULLETS = 20
SHIP_SIZE = 10
ASTEROID_MIN_SIZE = 20
ASTEROID_MAX_SIZE = 60
BULLET_SPEED = 13.0
SHIP_MAX_SPEED = 20
SHIP_ACCELERATION = 0.2
SHIP_TURN_SPEED = 20
ASTEROID_MIN_SPEED = 1.5
ASTEROID_MAX_SPEED = 4.5
HYPERSPACE_COOLDOWN = 10000
SHOOT_COOLDOWN = 200

def wrap_position(pos, surface_width, surface_height):
    x, y = pos
    return np.array([x % surface_width, y % surface_height], dtype=np.float32)

def get_random_velocity(min_speed, max_speed):
    speed = random.uniform(min_speed, max_speed)
    angle = random.uniform(0, 2 * math.pi)
    return np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32)

@dataclass
class GameObject:
    position: np.ndarray
    velocity: np.ndarray
    radius: float
    active: bool = True

    def move(self, dt: float, surface_width: int, surface_height: int):
        if self.active:
            self.position += self.velocity * dt * 60
            self.position = wrap_position(self.position, surface_width, surface_height)

    def collides_with(self, other) -> bool:
        if not (self.active and other.active):
            return False
        dx = self.position[0] - other.position[0]
        dy = self.position[1] - other.position[1]
        if abs(dx) > SCREEN_WIDTH / 2:
            dx = SCREEN_WIDTH - abs(dx)
            if self.position[0] < other.position[0]:
                dx = -dx
        if abs(dy) > SCREEN_HEIGHT / 2:
            dy = SCREEN_HEIGHT - abs(dy)
            if self.position[1] < other.position[1]:
                dy = -dy
        distance = math.sqrt(dx*dx + dy*dy)
        return distance < (self.radius + other.radius)

@dataclass
class Ship(GameObject):
    angle: float = 0.0
    thrust: bool = False
    in_hyperspace: bool = False
    hyperspace_ttl: int = 0
    direction: np.ndarray = None

    def __post_init__(self):
        if self.direction is None:
            self.direction = UP.copy()

    def rotate(self, clockwise: bool = True):
        multiplier = 1 if clockwise else -1
        self.angle += SHIP_TURN_SPEED * multiplier
        angle_rad = math.radians(self.angle)
        self.direction = np.array([math.sin(angle_rad), -math.cos(angle_rad)], dtype=np.float32)

    def accelerate(self):
        self.thrust = True
        thrust_vector = self.direction * SHIP_ACCELERATION
        self.velocity += thrust_vector
        speed = np.linalg.norm(self.velocity)
        if speed > SHIP_MAX_SPEED:
            self.velocity = (self.velocity / speed) * SHIP_MAX_SPEED

    def update(self, dt: float):
        if not self.active:
            return
        if not self.thrust:
            self.velocity *= 0.995
        if self.in_hyperspace:
            self.hyperspace_ttl -= 1
            if self.hyperspace_ttl <= 0:
                self.in_hyperspace = False
                self.position = np.array([
                    random.uniform(50, SCREEN_WIDTH - 50), 
                    random.uniform(50, SCREEN_HEIGHT - 50)
                ], dtype=np.float32)
                self.velocity = np.array([0, 0], dtype=np.float32)
        self.thrust = False

@dataclass  
class Asteroid(GameObject):
    size: int = 3
    rotation: float = 0.0
    rotation_speed: float = 0.0

    def __post_init__(self):
        self.rotation_speed = random.uniform(-2, 2)

    def update(self, dt: float):
        if self.active:
            self.rotation += self.rotation_speed

    def split(self) -> List['Asteroid']:
        if self.size <= 1:
            return []
        new_asteroids = []
        new_size = self.size - 1
        new_radius = ASTEROID_MIN_SIZE + (new_size - 1) * 15
        for i in range(2):
            angle = random.uniform(0, 2 * math.pi)
            speed_multiplier = 1.2
            base_speed = ASTEROID_MIN_SPEED * speed_multiplier
            max_speed = ASTEROID_MAX_SPEED * speed_multiplier
            new_velocity = get_random_velocity(base_speed, max_speed)
            new_asteroid = Asteroid(
                position=self.position.copy(),
                velocity=new_velocity,
                radius=new_radius,
                size=new_size
            )
            new_asteroids.append(new_asteroid)
        return new_asteroids

@dataclass
class Bullet(GameObject):
    lifetime: float = 180.0
    creation_time: int = 0

    def update(self, dt: float, current_time: int):
        if self.active and current_time - self.creation_time > self.lifetime:
            self.active = False

@dataclass
class Debris(GameObject):
    ttl: int = 50
    creation_time: int = 0

    def update(self, dt: float, current_time: int):
        if self.active and current_time - self.creation_time > self.ttl:
            self.active = False

class AsteroidsCore:
    WIDTH = SCREEN_WIDTH
    HEIGHT = SCREEN_HEIGHT

    def __init__(self, render_mode: str = "none", **kwargs):
        self.initial_asteroids = int(kwargs.pop("initial_asteroids", 4))
        self.max_lives = int(kwargs.pop("max_lives", 3))
        self.hyperspace_cooldown_time = int(kwargs.pop("hyperspace_cooldown", HYPERSPACE_COOLDOWN))
        self.shoot_cooldown_time = int(kwargs.pop("shoot_cooldown", SHOOT_COOLDOWN))
        self.score = 0
        self.lives = self.max_lives
        self.level = 1
        self.alive = True
        self.game_over = False
        self.frame_count = 0
        self.ship = None
        self.asteroids: List[Asteroid] = []
        self.bullets: List[Bullet] = []
        self.debris: List[Debris] = []
        self.shoot_timer = 0
        self.hyperspace_timer = 0
        self.respawn_timer = 0

        self.asteroids_destroyed_this_step = 0
        self.bullets_fired_this_step = 0
        self.collision_this_step = False
        self.hyperspace_used_this_step = False
        self.shots_hit_this_step = 0
        self.score_delta = 0
        self.last_score = 0
        self.ship_destroyed_this_step = False

        self.rng = np.random.RandomState(1337)
        obs_len = 6 + (MAX_ASTEROIDS * 5) + (MAX_BULLETS * 4) + 2
        self._obs_space = spaces.Box(-np.inf, np.inf, shape=(obs_len,), dtype=np.float32)
        self._act_space = spaces.Discrete(6)

        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        self._surf = pygame.Surface((self.WIDTH, self.HEIGHT))
        # --- Track targeting bonus history ---
        self.last_targeting_bonus = 0.0  # New: for change tracking
        self.reset()

    def get_action_space(self):
        return self._act_space

    def get_observation_space(self):
        return self._obs_space

    def _create_ship(self):
        position = np.array([SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2], dtype=np.float32)
        velocity = np.array([0, 0], dtype=np.float32)
        self.ship = Ship(position=position, velocity=velocity, radius=SHIP_SIZE)
        self.respawn_timer = 120

    def _spawn_asteroids(self, count: int):
        self.asteroids.clear()
        for _ in range(count):
            attempts = 0
            while attempts < 50:
                x = self.rng.uniform(0, SCREEN_WIDTH)
                y = self.rng.uniform(0, SCREEN_HEIGHT)
                if self.ship is None:
                    break
                dx = x - self.ship.position[0]
                dy = y - self.ship.position[1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > 150:
                    break
                attempts += 1
            position = np.array([x, y], dtype=np.float32)
            velocity = get_random_velocity(ASTEROID_MIN_SPEED, ASTEROID_MAX_SPEED)
            radius = ASTEROID_MIN_SIZE + (3 - 1) * 15
            asteroid = Asteroid(
                position=position,
                velocity=velocity,
                radius=radius,
                size=3
            )
            self.asteroids.append(asteroid)

    def _get_asteroids_sorted_by_distance(self) -> List:
        if not self.asteroids or not self.ship or not self.ship.active:
            return []
        distances_and_asteroids = []
        for asteroid in self.asteroids:
            if not asteroid.active:
                continue
            dx = self.ship.position[0] - asteroid.position[0]
            dy = self.ship.position[1] - asteroid.position[1]
            if abs(dx) > SCREEN_WIDTH / 2:
                dx = SCREEN_WIDTH - abs(dx)
                if self.ship.position[0] < asteroid.position[0]:
                    dx = -dx
            if abs(dy) > SCREEN_HEIGHT / 2:
                dy = SCREEN_HEIGHT - abs(dy)
                if self.ship.position[1] < asteroid.position[1]:
                    dy = -dy
            dist = math.sqrt(dx * dx + dy * dy)
            distances_and_asteroids.append((dist, asteroid))
        distances_and_asteroids.sort(key=lambda x: x[0])
        return distances_and_asteroids

    def _get_closest_asteroids_distances(self, n: int = 3) -> List[float]:
        sorted_asteroids = self._get_asteroids_sorted_by_distance()
        distances = [dist for dist, _ in sorted_asteroids]
        while len(distances) < n:
            distances.append(SCREEN_WIDTH)
        return distances[:n]

    def _calculate_targeting_info(self) -> tuple[float, float]:
        sorted_asteroids = self._get_asteroids_sorted_by_distance()
        if not sorted_asteroids:
            return 0.0, 180.0
        _, closest_ast = sorted_asteroids[0]
        ship_x, ship_y = self.ship.position
        ship_angle_deg = self.ship.angle % 360
        ship_angle_rad = math.radians(ship_angle_deg)
        dx = closest_ast.position[0] - ship_x
        dy = closest_ast.position[1] - ship_y
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0, 180.0
        ship_front = np.array([math.sin(ship_angle_rad), -math.cos(ship_angle_rad)])
        to_asteroid = np.array([dx, dy])
        normed_to_asteroid = to_asteroid / (np.linalg.norm(to_asteroid) + 1e-8)
        dot = np.clip(np.dot(ship_front, normed_to_asteroid), -1.0, 1.0)
        angle_diff_rad = math.acos(dot)
        angle_diff = math.degrees(angle_diff_rad)
        targeting_bonus = 4.0 * max(math.cos(angle_diff_rad), -0.125)
        return targeting_bonus, angle_diff

    def _obs(self) -> np.ndarray:
        obs = []
        if self.ship and self.ship.active:
            ship_obs = [
                self.ship.position[0] / SCREEN_WIDTH,
                self.ship.position[1] / SCREEN_HEIGHT,
                self.ship.velocity[0] / SHIP_MAX_SPEED,
                self.ship.velocity[1] / SHIP_MAX_SPEED,
                (self.ship.angle % 360) / 360.0,
                1.0
            ]
        else:
            ship_obs = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
        obs.extend(ship_obs)

        if self.ship and self.ship.active:
            distances_and_asteroids = self._get_asteroids_sorted_by_distance()
            sorted_asteroids = [ast for dist, ast in distances_and_asteroids[:MAX_ASTEROIDS]]
        else:
            sorted_asteroids = [a for a in self.asteroids if a.active][:MAX_ASTEROIDS]
        for i in range(MAX_ASTEROIDS):
            if i < len(sorted_asteroids):
                ast = sorted_asteroids[i]
                asteroid_obs = [
                    ast.position[0] / SCREEN_WIDTH,
                    ast.position[1] / SCREEN_HEIGHT,
                    ast.velocity[0] / ASTEROID_MAX_SPEED,
                    ast.velocity[1] / ASTEROID_MAX_SPEED,
                    ast.size / 3.0
                ]
            else:
                asteroid_obs = [0.0, 0.0, 0.0, 0.0, 0.0]
            obs.extend(asteroid_obs)
        active_bullets = [b for b in self.bullets if b.active][:MAX_BULLETS]
        for i in range(MAX_BULLETS):
            if i < len(active_bullets):
                bullet = active_bullets[i]
                bullet_obs = [
                    bullet.position[0] / SCREEN_WIDTH,
                    bullet.position[1] / SCREEN_HEIGHT,
                    bullet.velocity[0] / BULLET_SPEED,
                    bullet.velocity[1] / BULLET_SPEED
                ]
            else:
                bullet_obs = [0.0, 0.0, 0.0, 0.0]
            obs.extend(bullet_obs)
        # --- Add custom extras to obs ---
        obs.append(self.score / 10000.0)  # optionally normalize score if appropriate
        targeting_bonus, _ = self._calculate_targeting_info()
        obs.append(targeting_bonus)
        return np.array(obs, dtype=np.float32)

    def reset(self):
        self.score = 0
        self.lives = self.max_lives
        self.level = 1
        self.alive = True
        self.game_over = False
        self.frame_count = 0
        self._create_ship()
        self._spawn_asteroids(self.initial_asteroids)
        self.bullets.clear()
        self.debris.clear()
        self.shoot_timer = 0
        self.hyperspace_timer = 0
        self.respawn_timer = 0
        self.asteroids_destroyed_this_step = 0
        self.bullets_fired_this_step = 0
        self.collision_this_step = False
        self.hyperspace_used_this_step = False
        self.shots_hit_this_step = 0
        self.score_delta = 0
        self.last_score = 0
        self.ship_destroyed_this_step = False
        self.last_targeting_bonus = 0.0  # <-- Reset this on episode start!
        return self._obs()

    def step(self, action: int):
        if not self.alive:
            return self._obs(), 0.0, True, {"episode_end": True, "won": False}

        dt = 1.0 / 60.0
        self.frame_count += 1

        self.asteroids_destroyed_this_step = 0
        self.bullets_fired_this_step = 0
        self.collision_this_step = False
        self.hyperspace_used_this_step = False
        self.shots_hit_this_step = 0
        self.ship_destroyed_this_step = False
        self.shoot_timer = max(0, self.shoot_timer - 1)
        self.hyperspace_timer = max(0, self.hyperspace_timer - 1)
        self.respawn_timer = max(0, self.respawn_timer - 1)

        a = int(action)
        if self.ship and self.ship.active and not self.ship.in_hyperspace:
            if a == 1:
                self.ship.rotate(clockwise=False)
            elif a == 2:
                self.ship.rotate(clockwise=True)
            elif a == 3:
                self.ship.accelerate()
            elif a == 4:
                if self.shoot_timer <= 0 and len([b for b in self.bullets if b.active]) < MAX_BULLETS:
                    bullet_pos = self.ship.position + self.ship.direction * (SHIP_SIZE + 5)
                    bullet_vel = self.ship.direction * BULLET_SPEED + self.ship.velocity
                    bullet = Bullet(
                        position=bullet_pos,
                        velocity=bullet_vel,
                        radius=2,
                        creation_time=self.frame_count
                    )
                    self.bullets.append(bullet)
                    self.shoot_timer = self.shoot_cooldown_time // 17
                    self.bullets_fired_this_step = 1
            elif a == 5:
                if self.hyperspace_timer <= 0:
                    self.ship.in_hyperspace = True
                    self.ship.hyperspace_ttl = 100
                    self.hyperspace_timer = self.hyperspace_cooldown_time // 17
                    self.hyperspace_used_this_step = True

        if self.ship:
            self.ship.update(dt)
            if not self.ship.in_hyperspace:
                self.ship.move(dt, SCREEN_WIDTH, SCREEN_HEIGHT)

        for asteroid in self.asteroids:
            asteroid.update(dt)
            asteroid.move(dt, SCREEN_WIDTH, SCREEN_HEIGHT)
        for bullet in self.bullets:
            bullet.update(dt, self.frame_count)
            if bullet.active:
                bullet.move(dt, SCREEN_WIDTH, SCREEN_HEIGHT)
        for debris_piece in self.debris:
            debris_piece.update(dt, self.frame_count)
            if debris_piece.active:
                debris_piece.move(dt, SCREEN_WIDTH, SCREEN_HEIGHT)

        self.bullets = [b for b in self.bullets if b.active]
        self.debris = [d for d in self.debris if d.active]
        self._check_collisions()
        if not self.asteroids and self.alive:
            self.level += 1
            self.score += 100
            self._spawn_asteroids(self.initial_asteroids + self.level - 1)
        if self.lives <= 0:
            self.alive = False
            self.game_over = True
        self.score_delta = self.score - self.last_score
        self.last_score = self.score

        closest_3_distances = self._get_closest_asteroids_distances(3)
        nearest_distance = closest_3_distances[0]
        targeting_bonus, angle_diff = self._calculate_targeting_info()
        # --- Track the change in targeting bonus! ---
        targeting_bonus_delta = targeting_bonus - self.last_targeting_bonus
        self.last_targeting_bonus = targeting_bonus

        base_reward = 1.0
        terminated = not self.alive

        info = {
            "score_delta": self.score_delta,
            "score": self.score,
            "lives": self.lives,
            "level": self.level,
            "collision": self.collision_this_step,
            "hyperspace_used": self.hyperspace_used_this_step,
            "bullets_fired": self.bullets_fired_this_step,
            "ship_destroyed": self.ship_destroyed_this_step,
            "asteroids_destroyed": self.asteroids_destroyed_this_step,
            "distance_to_nearest": nearest_distance,
            "distances_to_closest_3": closest_3_distances,
            "avg_distance_closest_3": sum(closest_3_distances) / len(closest_3_distances),
            "targeting_bonus": targeting_bonus,
            "targeting_bonus_delta": targeting_bonus_delta,  # <-- NEW METRIC!
            "angle_to_nearest": angle_diff,
            "asteroids_remaining": len([a for a in self.asteroids if a.active]),
            "bullets_active": len([b for b in self.bullets if b.active]),
            "ship_speed": np.linalg.norm(self.ship.velocity) if self.ship and self.ship.active else 0.0,
            "ship_angle": self.ship.angle if self.ship else 0.0,
            "in_hyperspace": self.ship.in_hyperspace if self.ship else False,
            "threat_level": len(self.asteroids),
            "episode_end": terminated,
            "won": False,
            "level_completed": len(self.asteroids) == 0 and self.alive
        }
        return self._obs(), float(base_reward), bool(terminated), info

    def _check_collisions(self):
        if (self.ship and self.ship.active and self.respawn_timer <= 0 and 
            not self.ship.in_hyperspace):
            for asteroid in self.asteroids:
                if asteroid.active and self.ship.collides_with(asteroid):
                    self._destroy_ship()
                    break
        for bullet in self.bullets[:]:
            if not bullet.active:
                continue
            for asteroid in self.asteroids[:]:
                if not asteroid.active:
                    continue
                if bullet.collides_with(asteroid):
                    bullet.active = False
                    self._destroy_asteroid(asteroid)
                    self.shots_hit_this_step += 1
                    break

    def _destroy_ship(self):
        if self.ship:
            self.ship_destroyed_this_step = True
            self.collision_this_step = True
            self.lives -= 1
            for _ in range(8):
                debris_piece = Debris(
                    position=self.ship.position.copy() + np.random.uniform(-10, 10, 2),
                    velocity=np.random.uniform(-3, 3, 2),
                    radius=1,
                    creation_time=self.frame_count
                )
                self.debris.append(debris_piece)
            if self.lives > 0:
                self._create_ship()
            else:
                self.ship = None

    def _destroy_asteroid(self, asteroid: Asteroid):
        self.asteroids_destroyed_this_step += 1
        score_values = {3: 20, 2: 50, 1: 100}
        self.score += score_values.get(asteroid.size, 20)
        for _ in range(4):
            debris_piece = Debris(
                position=asteroid.position.copy() + np.random.uniform(-5, 5, 2),
                velocity=np.random.uniform(-2, 2, 2),
                radius=1,
                creation_time=self.frame_count
            )
            self.debris.append(debris_piece)
        new_asteroids = asteroid.split()
        self.asteroids.extend(new_asteroids)
        asteroid.active = False
        self.asteroids = [a for a in self.asteroids if a.active]

    def render(self, surface: pygame.Surface, blit_only: bool = True):
        surface.fill((0, 0, 0))
        for asteroid in self.asteroids:
            if not asteroid.active:
                continue
            color = (200, 200, 200)
            center = (int(asteroid.position[0]), int(asteroid.position[1]))
            radius = int(asteroid.radius)
            pygame.draw.circle(surface, color, center, radius, 2)
            angle_rad = math.radians(asteroid.rotation)
            end_x = center[0] + math.cos(angle_rad) * radius * 0.8
            end_y = center[1] + math.sin(angle_rad) * radius * 0.8
            pygame.draw.line(surface, color, center, (int(end_x), int(end_y)), 1)
        for bullet in self.bullets:
            if bullet.active:
                center = (int(bullet.position[0]), int(bullet.position[1]))
                pygame.draw.circle(surface, (255, 255, 255), center, 2)
        for debris_piece in self.debris:
            if debris_piece.active:
                center = (int(debris_piece.position[0]), int(debris_piece.position[1]))
                pygame.draw.circle(surface, (150, 150, 150), center, 1)
        if self.ship and self.ship.active:
            draw_ship = True
            if self.respawn_timer > 0:
                draw_ship = (self.frame_count // 5) % 2 == 0
            if self.ship.in_hyperspace:
                draw_ship = False
            if draw_ship:
                center = self.ship.position
                angle_rad = math.radians(self.ship.angle)
                nose = center + np.array([
                    math.sin(angle_rad) * SHIP_SIZE,
                    -math.cos(angle_rad) * SHIP_SIZE
                ])
                left_rear = center + np.array([
                    math.sin(angle_rad + 2.5) * SHIP_SIZE * 0.7,
                    -math.cos(angle_rad + 2.5) * SHIP_SIZE * 0.7
                ])
                right_rear = center + np.array([
                    math.sin(angle_rad - 2.5) * SHIP_SIZE * 0.7,
                    -math.cos(angle_rad - 2.5) * SHIP_SIZE * 0.7
                ])
                points = [
                    (int(nose[0]), int(nose[1])),
                    (int(left_rear[0]), int(left_rear[1])),
                    (int(right_rear[0]), int(right_rear[1]))
                ]
                pygame.draw.polygon(surface, (255, 255, 255), points, 2)
                if self.ship.thrust:
                    thrust_end = center - np.array([
                        math.sin(angle_rad) * SHIP_SIZE * 1.5,
                        -math.cos(angle_rad) * SHIP_SIZE * 1.5
                    ])
                    thrust_points = [
                        (int(left_rear[0]), int(left_rear[1])),
                        (int(thrust_end[0]), int(thrust_end[1])),
                        (int(right_rear[0]), int(right_rear[1]))
                    ]
                    pygame.draw.polygon(surface, (255, 100, 100), thrust_points)
            if self.ship and self.ship.active and any(a.active for a in self.asteroids):
                distances_and_asteroids = self._get_asteroids_sorted_by_distance()
                ship_x, ship_y = self.ship.position
                # debug_font = pygame.font.Font(None, 32)
                # for idx in range(min(3, len(distances_and_asteroids))):
                #     dist, asteroid = distances_and_asteroids[idx]
                #     ast_x, ast_y = asteroid.position
                #     if idx == 0:
                #         color = (0, 255, 0)
                #     elif idx == 1:
                #         color = (255, 255, 0)
                #     else:
                #         color = (0, 255, 255)
                #     pygame.draw.line(surface, color, (int(ship_x), int(ship_y)), (int(ast_x), int(ast_y)), 2)
                # targeting_bonus, angle_diff = self._calculate_targeting_info()
                # angle_text = debug_font.render(f"Angle: {angle_diff:.1f}°", True, (0, 255, 0))
                # target_text = debug_font.render(f"TargetBonus: {targeting_bonus:.2f}", True, (0, 255, 255))
                # surface.blit(angle_text, (int(ship_x) + 20, int(ship_y) - 30))
                # surface.blit(target_text, (int(ship_x) + 20, int(ship_y) - 60))
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"SCORE: {self.score:06d}", True, (255, 255, 255))
        surface.blit(score_text, (10, 10))
        lives_text = font.render(f"LIVES: {max(0, self.lives)}", True, (255, 255, 255))
        surface.blit(lives_text, (10, 50))
        level_text = font.render(f"LEVEL: {self.level}", True, (255, 255, 255))
        surface.blit(level_text, (10, 90))
        if self.game_over:
            game_over_text = font.render("GAME OVER", True, (255, 0, 0))
            text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
            surface.blit(game_over_text, text_rect)
