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
    """
    Wrap game object position around screen edges (screen wrapping).
    Objects that move past one edge appear on the opposite edge.
    
    Args:
        pos: Current position [x, y]
        surface_width: Screen width in pixels
        surface_height: Screen height in pixels
    
    Returns:
        np.ndarray: Wrapped position within screen bounds
    """
    x, y = pos
    return np.array([x % surface_width, y % surface_height], dtype=np.float32)

def get_random_velocity(min_speed, max_speed):
    """
    Generate a random velocity vector with speed between min_speed and max_speed.
    Direction is random (0-360 degrees).
    
    Args:
        min_speed: Minimum speed magnitude
        max_speed: Maximum speed magnitude
    
    Returns:
        np.ndarray: Velocity vector [vx, vy]
    """
    speed = random.uniform(min_speed, max_speed)
    angle = random.uniform(0, 2 * math.pi)
    return np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32)

@dataclass
class GameObject:
    """
    Base class for all game objects (ship, asteroids, bullets, debris).
    Handles position, velocity, collision detection, and movement.
    """
    position: np.ndarray
    velocity: np.ndarray
    radius: float
    active: bool = True

    def move(self, dt: float, surface_width: int, surface_height: int):
        """
        Update object position based on velocity and handle screen wrapping.
        
        Args:
            dt: Delta time (frame time)
            surface_width: Screen width for wrapping
            surface_height: Screen height for wrapping
        """
        if self.active:
            self.position += self.velocity * dt * 60
            self.position = wrap_position(self.position, surface_width, surface_height)

    def collides_with(self, other) -> bool:
        """
        Check if this object collides with another object using circular collision.
        Handles screen wrapping by checking shortest distance across edges.
        
        Args:
            other: Another GameObject to check collision with
        
        Returns:
            bool: True if objects are colliding
        """
        if not (self.active and other.active):
            return False
        
        dx = self.position[0] - other.position[0]
        dy = self.position[1] - other.position[1]
        
        # Handle screen wrapping - use shortest distance across edges
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
    """
    Player ship class. Handles rotation, thrust, hyperspace, and ship-specific behavior.
    """
    angle: float = 0.0
    thrust: bool = False
    in_hyperspace: bool = False
    hyperspace_ttl: int = 0
    direction: np.ndarray = None
    
    def __post_init__(self):
        """Initialize ship direction vector if not provided."""
        if self.direction is None:
            self.direction = UP.copy()
    
    def rotate(self, clockwise: bool = True):
        """
        Rotate the ship and update its direction vector.
        
        Args:
            clockwise: True to rotate clockwise, False for counter-clockwise
        """
        multiplier = 1 if clockwise else -1
        self.angle += SHIP_TURN_SPEED * multiplier
        angle_rad = math.radians(self.angle)
        self.direction = np.array([math.sin(angle_rad), -math.cos(angle_rad)], dtype=np.float32)
    
    def accelerate(self):
        """
        Apply thrust in the ship's current direction.
        Adds thrust vector to velocity and caps at maximum speed.
        """
        self.thrust = True
        thrust_vector = self.direction * SHIP_ACCELERATION
        self.velocity += thrust_vector
        speed = np.linalg.norm(self.velocity)
        if speed > SHIP_MAX_SPEED:
            self.velocity = (self.velocity / speed) * SHIP_MAX_SPEED
    
    def update(self, dt: float):
        """
        Update ship state each frame.
        Handles thrust decay, hyperspace teleportation, and state management.
        
        Args:
            dt: Delta time (frame time)
        """
        if not self.active:
            return
        
        # Apply drag when not thrusting
        if not self.thrust:
            self.velocity *= 0.995
        
        # Handle hyperspace teleportation
        if self.in_hyperspace:
            self.hyperspace_ttl -= 1
            if self.hyperspace_ttl <= 0:
                self.in_hyperspace = False
                # Teleport to random safe position
                self.position = np.array([
                    random.uniform(50, SCREEN_WIDTH - 50), 
                    random.uniform(50, SCREEN_HEIGHT - 50)
                ], dtype=np.float32)
                self.velocity = np.array([0, 0], dtype=np.float32)
        
        # Reset thrust flag (must be set each frame)
        self.thrust = False

@dataclass  
class Asteroid(GameObject):
    """
    Asteroid class. Handles rotation, splitting when destroyed, and asteroid-specific behavior.
    """
    size: int = 3  # Size 3=large, 2=medium, 1=small
    rotation: float = 0.0
    rotation_speed: float = 0.0
    
    def __post_init__(self):
        """Initialize random rotation speed for visual variety."""
        self.rotation_speed = random.uniform(-2, 2)
    
    def update(self, dt: float):
        """
        Update asteroid rotation each frame.
        
        Args:
            dt: Delta time (frame time)
        """
        if self.active:
            self.rotation += self.rotation_speed
    
    def split(self) -> List['Asteroid']:
        """
        Split asteroid into two smaller asteroids when destroyed.
        Size 1 asteroids don't split (fully destroyed).
        
        Returns:
            List[Asteroid]: List of new smaller asteroids (empty if size 1)
        """
        if self.size <= 1:
            return []
        
        new_asteroids = []
        new_size = self.size - 1
        new_radius = ASTEROID_MIN_SIZE + (new_size - 1) * 15
        
        # Create two smaller asteroids with random velocities
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
    """
    Bullet class. Handles bullet lifetime and automatic cleanup.
    """
    lifetime: float = 3600.0  # Frames until bullet expires
    creation_time: int = 0
    
    def update(self, dt: float, current_time: int):
        """
        Update bullet and check if it should expire.
        
        Args:
            dt: Delta time (frame time)
            current_time: Current game frame count
        """
        if self.active and current_time - self.creation_time > self.lifetime:
            self.active = False

@dataclass
class Debris(GameObject):
    """
    Debris particles created when objects are destroyed. Purely visual.
    """
    ttl: int = 50  # Time to live in frames
    creation_time: int = 0
    
    def update(self, dt: float, current_time: int):
        """
        Update debris and check if it should disappear.
        
        Args:
            dt: Delta time (frame time)
            current_time: Current game frame count
        """
        if self.active and current_time - self.creation_time > self.ttl:
            self.active = False

class AsteroidsCore:
    """
    Main Asteroids game engine. Handles game logic, collision detection,
    scoring, level progression, and provides Gymnasium environment interface.
    """
    WIDTH = SCREEN_WIDTH
    HEIGHT = SCREEN_HEIGHT
    
    def __init__(self, render_mode: str = "none", **kwargs):
        """
        Initialize the Asteroids game core.
        
        Args:
            render_mode: Rendering mode ("none", "human", etc.)
            **kwargs: Game configuration (initial_asteroids, max_lives, etc.)
        """
        # Game configuration
        self.initial_asteroids = int(kwargs.pop("initial_asteroids", 4))
        self.max_lives = int(kwargs.pop("max_lives", 3))
        self.hyperspace_cooldown_time = int(kwargs.pop("hyperspace_cooldown", HYPERSPACE_COOLDOWN))
        self.shoot_cooldown_time = int(kwargs.pop("shoot_cooldown", SHOOT_COOLDOWN))
        
        # Game state
        self.score = 0
        self.lives = self.max_lives
        self.level = 1
        self.alive = True
        self.game_over = False
        self.frame_count = 0
        
        # Game objects
        self.ship = None
        self.asteroids: List[Asteroid] = []
        self.bullets: List[Bullet] = []
        self.debris: List[Debris] = []
        
        # Timers
        self.shoot_timer = 0
        self.hyperspace_timer = 0
        self.respawn_timer = 0
        
        # Step tracking variables for info dict
        self.asteroids_destroyed_this_step = 0
        self.bullets_fired_this_step = 0
        self.collision_this_step = False
        self.hyperspace_used_this_step = False
        self.shots_hit_this_step = 0
        self.score_delta = 0
        self.last_score = 0
        self.ship_destroyed_this_step = False
        
        # RL interface setup
        self.rng = np.random.RandomState(1337)
        obs_len = 6 + (MAX_ASTEROIDS * 5) + (MAX_BULLETS * 4)
        self._obs_space = spaces.Box(-np.inf, np.inf, shape=(obs_len,), dtype=np.float32)
        self._act_space = spaces.Discrete(6)
        
        # Pygame setup
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        self._surf = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.reset()
    
    def get_action_space(self):
        """Return the Gymnasium action space (6 discrete actions)."""
        return self._act_space
    
    def get_observation_space(self):
        """Return the Gymnasium observation space."""
        return self._obs_space
    
    def _create_ship(self):
        """Create a new ship at the center of the screen with respawn timer."""
        position = np.array([SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2], dtype=np.float32)
        velocity = np.array([0, 0], dtype=np.float32)
        self.ship = Ship(position=position, velocity=velocity, radius=SHIP_SIZE)
        self.respawn_timer = 120  # Invulnerability frames after respawn
    
    def _spawn_asteroids(self, count: int):
        """
        Spawn initial asteroids for a new level.
        Ensures asteroids spawn away from the ship to avoid immediate collision.
        
        Args:
            count: Number of asteroids to spawn
        """
        self.asteroids.clear()
        for _ in range(count):
            attempts = 0
            # Try to find a safe spawn position away from ship
            while attempts < 50:
                x = self.rng.uniform(0, SCREEN_WIDTH)
                y = self.rng.uniform(0, SCREEN_HEIGHT)
                if self.ship is None:
                    break
                dx = x - self.ship.position[0]
                dy = y - self.ship.position[1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > 150:  # Safe distance from ship
                    break
                attempts += 1
            
            position = np.array([x, y], dtype=np.float32)
            velocity = get_random_velocity(ASTEROID_MIN_SPEED, ASTEROID_MAX_SPEED)
            radius = ASTEROID_MIN_SIZE + (3 - 1) * 15  # Large asteroid
            asteroid = Asteroid(
                position=position,
                velocity=velocity,
                radius=radius,
                size=3
            )
            self.asteroids.append(asteroid)
    
    def _get_asteroids_sorted_by_distance(self) -> List[Tuple[float, 'Asteroid']]:
        """
        Get all active asteroids sorted by distance from the ship.
        Returns a list of (distance, asteroid) tuples sorted by distance (closest first).
        Returns empty list if no ship or no asteroids.
        """
        if not self.asteroids or not self.ship or not self.ship.active:
            return []

        distances_and_asteroids = []
        for asteroid in self.asteroids:
            if not asteroid.active:
                continue
            dx = self.ship.position[0] - asteroid.position[0]
            dy = self.ship.position[1] - asteroid.position[1]
            # Handle screen wrapping
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
        """
        Get distances to the N closest asteroids from the ship.
        Returns: List[float]: Distances to N closest asteroids (padded with SCREEN_WIDTH if needed)
        """
        sorted_asteroids = self._get_asteroids_sorted_by_distance()
        distances = [dist for dist, _ in sorted_asteroids]
        while len(distances) < n:
            distances.append(SCREEN_WIDTH)
        return distances[:n]

    # def _calculate_targeting_info(self) -> tuple[float, float]:
    #     """
    #     Calculate targeting bonus and angle difference to nearest asteroid.
    #     Returns: (targeting_bonus, angle_difference)
    #     """
    #     sorted_asteroids = self._get_asteroids_sorted_by_distance()
    #     if not sorted_asteroids:
    #         return 0.0, 180.0
    #     _, closest_ast = sorted_asteroids[0]
    #     ship_x, ship_y = self.ship.position
    #     ship_angle = self.ship.angle % 360

    #     dx = closest_ast.position[0] - ship_x
    #     dy = closest_ast.position[1] - ship_y
    #     if abs(dx) < 1e-6 and abs(dy) < 1e-6:
    #         return 0.0, 180.0

    #     # Subtract 90 degrees so 0 points to ship's front ('up')
    #     target_angle = (math.degrees(math.atan2(dy, dx)) - 90) % 360
    #     angle_diff = abs((ship_angle - target_angle + 180) % 360 - 180)

    #     while angle_diff > 180:
    #         angle_diff = abs(angle_diff - 360)
    #     targeting_bonus = 4.0 * max(math.cos(math.radians(angle_diff)), -0.125)
    #     return targeting_bonus, angle_diff
        

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

        # Ship's facing direction (unit vector!)
        ship_front = np.array([math.sin(ship_angle_rad), -math.cos(ship_angle_rad)])
        to_asteroid = np.array([dx, dy])
        normed_to_asteroid = to_asteroid / (np.linalg.norm(to_asteroid) + 1e-8)

        # Compute angle difference
        dot = np.clip(np.dot(ship_front, normed_to_asteroid), -1.0, 1.0)
        angle_diff_rad = math.acos(dot)
        angle_diff = math.degrees(angle_diff_rad)

        # Targeting bonus (use full cosine)
        targeting_bonus = 4.0 * max(dot, -0.125)
        return targeting_bonus, angle_diff


    def _obs(self) -> np.ndarray:
        """
        Generate observation vector for RL agent.
        Includes ship state, asteroid states (sorted by distance), and bullet states.

        Returns:
            np.ndarray: Flattened observation vector with normalized values
        """
        obs = []
    
        # Ship observation (6 values)
        if self.ship and self.ship.active:
            ship_obs = [
                self.ship.position[0] / SCREEN_WIDTH,      # x position (0-1)
                self.ship.position[1] / SCREEN_HEIGHT,     # y position (0-1)
                self.ship.velocity[0] / SHIP_MAX_SPEED,    # x velocity (-1 to 1)
                self.ship.velocity[1] / SHIP_MAX_SPEED,    # y velocity (-1 to 1)
                (self.ship.angle % 360) / 360.0,           # angle (0-1)
                1.0                                        # alive flag
            ]
        else:
            ship_obs = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0]    # Dead ship defaults
        obs.extend(ship_obs)
    
        # Asteroid observations (MAX_ASTEROIDS * 5 values, sorted by distance)
        if self.ship and self.ship.active:
            distances_and_asteroids = self._get_asteroids_sorted_by_distance()
            sorted_asteroids = [ast for dist, ast in distances_and_asteroids[:MAX_ASTEROIDS]]
        else:
            sorted_asteroids = [a for a in self.asteroids if a.active][:MAX_ASTEROIDS]
    
        # Add asteroid observations (5 values each)
        for i in range(MAX_ASTEROIDS):
            if i < len(sorted_asteroids):
                ast = sorted_asteroids[i]
                asteroid_obs = [
                    ast.position[0] / SCREEN_WIDTH,        # x position (0-1)
                    ast.position[1] / SCREEN_HEIGHT,       # y position (0-1)
                    ast.velocity[0] / ASTEROID_MAX_SPEED,  # x velocity
                    ast.velocity[1] / ASTEROID_MAX_SPEED,  # y velocity
                    ast.size / 3.0                         # size (0.33, 0.67, 1.0)
                ]
            else:
                asteroid_obs = [0.0, 0.0, 0.0, 0.0, 0.0]  # Empty slot
            obs.extend(asteroid_obs)
    
        # Bullet observations (MAX_BULLETS * 4 values)
        active_bullets = [b for b in self.bullets if b.active][:MAX_BULLETS]
        for i in range(MAX_BULLETS):
            if i < len(active_bullets):
                bullet = active_bullets[i]
                bullet_obs = [
                    bullet.position[0] / SCREEN_WIDTH,     # x position (0-1)
                    bullet.position[1] / SCREEN_HEIGHT,    # y position (0-1)
                    bullet.velocity[0] / BULLET_SPEED,     # x velocity
                    bullet.velocity[1] / BULLET_SPEED      # y velocity
                ]
            else:
                bullet_obs = [0.0, 0.0, 0.0, 0.0]        # Empty slot
            obs.extend(bullet_obs)
    
        return np.array(obs, dtype=np.float32)

    
    def reset(self):
        """
        Reset game to initial state for new episode.
        Clears all objects and resets game variables.
        
        Returns:
            np.ndarray: Initial observation
        """
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
        
        # Reset step tracking variables
        self.asteroids_destroyed_this_step = 0
        self.bullets_fired_this_step = 0
        self.collision_this_step = False
        self.hyperspace_used_this_step = False
        self.shots_hit_this_step = 0
        self.score_delta = 0
        self.last_score = 0
        self.ship_destroyed_this_step = False
        
        return self._obs()
    
    def step(self, action: int):
        """
        Execute one game step with the given action.
        Updates all game objects, handles collisions, and returns RL step info.
        
        Args:
            action: Integer action (0=noop, 1=left, 2=right, 3=thrust, 4=shoot, 5=hyperspace)
        
        Returns:
            tuple: (observation, reward, terminated, info)
        """
        if not self.alive:
            return self._obs(), 0.0, True, {"episode_end": True, "won": False}
        
        dt = 1.0 / 60.0
        self.frame_count += 1
        
        # Reset step tracking variables
        self.asteroids_destroyed_this_step = 0
        self.bullets_fired_this_step = 0
        self.collision_this_step = False
        self.hyperspace_used_this_step = False
        self.shots_hit_this_step = 0
        self.ship_destroyed_this_step = False
        
        # Update cooldown timers
        self.shoot_timer = max(0, self.shoot_timer - 1)
        self.hyperspace_timer = max(0, self.hyperspace_timer - 1)
        self.respawn_timer = max(0, self.respawn_timer - 1)
        
        # Handle player actions
        a = int(action)
        if self.ship and self.ship.active and not self.ship.in_hyperspace:
            if a == 1:  # Turn left
                self.ship.rotate(clockwise=False)
            elif a == 2:  # Turn right
                self.ship.rotate(clockwise=True)
            elif a == 3:  # Thrust
                self.ship.accelerate()
            elif a == 4:  # Shoot
                if self.shoot_timer <= 0 and len([b for b in self.bullets if b.active]) < MAX_BULLETS:
                    # Create bullet in front of ship
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
            elif a == 5:  # Hyperspace
                if self.hyperspace_timer <= 0:
                    self.ship.in_hyperspace = True
                    self.ship.hyperspace_ttl = 100
                    self.hyperspace_timer = self.hyperspace_cooldown_time // 17
                    self.hyperspace_used_this_step = True
        
        # Update all game objects
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
        
        # Clean up inactive objects
        self.bullets = [b for b in self.bullets if b.active]
        self.debris = [d for d in self.debris if d.active]
        
        # Check all collisions
        self._check_collisions()
        
        # Check for level completion
        if not self.asteroids and self.alive:
            self.level += 1
            self.score += 100  # Level completion bonus
            self._spawn_asteroids(self.initial_asteroids + self.level - 1)
        
        # Check for game over
        if self.lives <= 0:
            self.alive = False
            self.game_over = True
        
        # Calculate score change
        self.score_delta = self.score - self.last_score
        self.last_score = self.score
        
        # Calculate all spatial and targeting info once per step
        closest_3_distances = self._get_closest_asteroids_distances(3)
        nearest_distance = closest_3_distances[0]
        targeting_bonus, angle_diff = self._calculate_targeting_info()
        
        # Prepare step return values
        base_reward = 0.1  # Small per-frame survival reward
        terminated = not self.alive
        
        # Comprehensive info dict for reward functions
        info = {
            # Score and progression
            "score_delta": self.score_delta,
            "score": self.score,
            "lives": self.lives,
            "level": self.level,
            
            # Actions and events this step
            "collision": self.collision_this_step,
            "hyperspace_used": self.hyperspace_used_this_step,
            "bullets_fired": self.bullets_fired_this_step,
            "shots_hit": self.shots_hit_this_step,
            "ship_destroyed": self.ship_destroyed_this_step,
            "asteroids_destroyed": self.asteroids_destroyed_this_step,
            
            # Spatial information (calculated once)
            "distance_to_nearest": nearest_distance,
            "distances_to_closest_3": closest_3_distances,
            "avg_distance_closest_3": sum(closest_3_distances) / len(closest_3_distances),
            
            # Targeting information (calculated once)
            "targeting_bonus": targeting_bonus,
            "angle_to_nearest": angle_diff,
            
            # Game state
            "asteroids_remaining": len([a for a in self.asteroids if a.active]),
            "bullets_active": len([b for b in self.bullets if b.active]),
            "ship_speed": np.linalg.norm(self.ship.velocity) if self.ship and self.ship.active else 0.0,
            "ship_angle": self.ship.angle if self.ship else 0.0,
            "in_hyperspace": self.ship.in_hyperspace if self.ship else False,
            
            # Calculated metrics
            "accuracy": self.shots_hit_this_step / max(1, self.bullets_fired_this_step) if self.bullets_fired_this_step > 0 else 0.0,
            "threat_level": len(self.asteroids),
            
            # Episode status
            "episode_end": terminated,
            "won": False,
            "level_completed": len(self.asteroids) == 0 and self.alive
        }
        
        return self._obs(), float(base_reward), bool(terminated), info
    
    def _check_collisions(self):
        """
        Check and handle all collision types:
        1. Ship-asteroid collisions (destroy ship)
        2. Bullet-asteroid collisions (destroy asteroid, remove bullet)
        """
        # Ship-asteroid collisions
        if (self.ship and self.ship.active and self.respawn_timer <= 0 and 
            not self.ship.in_hyperspace):
            for asteroid in self.asteroids:
                if asteroid.active and self.ship.collides_with(asteroid):
                    self._destroy_ship()
                    break
        
        # Bullet-asteroid collisions
        for bullet in self.bullets[:]:  # Copy list to avoid modification during iteration
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
        """
        Handle ship destruction.
        Creates debris, reduces lives, respawns ship if lives remain.
        """
        if self.ship:
            self.ship_destroyed_this_step = True
            self.collision_this_step = True
            self.lives -= 1
            
            # Create visual debris
            for _ in range(8):
                debris_piece = Debris(
                    position=self.ship.position.copy() + np.random.uniform(-10, 10, 2),
                    velocity=np.random.uniform(-3, 3, 2),
                    radius=1,
                    creation_time=self.frame_count
                )
                self.debris.append(debris_piece)
            
            # Respawn or remove ship
            if self.lives > 0:
                self._create_ship()
            else:
                self.ship = None
    
    def _destroy_asteroid(self, asteroid: Asteroid):
        """
        Handle asteroid destruction.
        Awards points, creates debris, splits asteroid into smaller pieces.
        
        Args:
            asteroid: The asteroid to destroy
        """
        self.asteroids_destroyed_this_step += 1
        
        # Award points based on asteroid size
        score_values = {3: 20, 2: 50, 1: 100}  # Smaller = more points
        self.score += score_values.get(asteroid.size, 20)
        
        # Create visual debris
        for _ in range(4):
            debris_piece = Debris(
                position=asteroid.position.copy() + np.random.uniform(-5, 5, 2),
                velocity=np.random.uniform(-2, 2, 2),
                radius=1,
                creation_time=self.frame_count
            )
            self.debris.append(debris_piece)
        
        # Split into smaller asteroids (if not already smallest)
        new_asteroids = asteroid.split()
        self.asteroids.extend(new_asteroids)
        
        # Remove the destroyed asteroid
        asteroid.active = False
        self.asteroids = [a for a in self.asteroids if a.active]
    
    def render(self, surface: pygame.Surface, blit_only: bool = True):
        """
        Render the game to a pygame surface.
        Draws all game objects, HUD, and debug information.
        
        Args:
            surface: Pygame surface to draw on
            blit_only: If True, only blit to surface (for compatibility)
        """
        # Clear screen
        surface.fill((0, 0, 0))
        
        # Draw asteroids
        for asteroid in self.asteroids:
            if not asteroid.active:
                continue
            color = (200, 200, 200)
            center = (int(asteroid.position[0]), int(asteroid.position[1]))
            radius = int(asteroid.radius)
            pygame.draw.circle(surface, color, center, radius, 2)
            
            # Draw rotation indicator line
            angle_rad = math.radians(asteroid.rotation)
            end_x = center[0] + math.cos(angle_rad) * radius * 0.8
            end_y = center[1] + math.sin(angle_rad) * radius * 0.8
            pygame.draw.line(surface, color, center, (int(end_x), int(end_y)), 1)
        
        # Draw bullets
        for bullet in self.bullets:
            if bullet.active:
                center = (int(bullet.position[0]), int(bullet.position[1]))
                pygame.draw.circle(surface, (255, 255, 255), center, 2)
        
        # Draw debris particles
        for debris_piece in self.debris:
            if debris_piece.active:
                center = (int(debris_piece.position[0]), int(debris_piece.position[1]))
                pygame.draw.circle(surface, (150, 150, 150), center, 1)
        
        # Draw ship
        if self.ship and self.ship.active:
            draw_ship = True
            
            # Flashing during respawn invulnerability
            if self.respawn_timer > 0:
                draw_ship = (self.frame_count // 5) % 2 == 0
            
            # Hidden during hyperspace
            if self.ship.in_hyperspace:
                draw_ship = False
            
            if draw_ship:
                center = self.ship.position
                angle_rad = math.radians(self.ship.angle)
                
                # Ship triangle vertices
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
                
                # Draw ship triangle
                points = [
                    (int(nose[0]), int(nose[1])),
                    (int(left_rear[0]), int(left_rear[1])),
                    (int(right_rear[0]), int(right_rear[1]))
                ]
                pygame.draw.polygon(surface, (255, 255, 255), points, 2)
                
                # Draw thrust flame when thrusting
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
            # Draw lines to the 3 closest asteroids (debug visualization)
            if self.ship and self.ship.active and any(a.active for a in self.asteroids):
                distances_and_asteroids = self._get_asteroids_sorted_by_distance()
                ship_x, ship_y = self.ship.position
                debug_font = pygame.font.Font(None, 32)

                for idx in range(min(3, len(distances_and_asteroids))):
                    dist, asteroid = distances_and_asteroids[idx]
                    ast_x, ast_y = asteroid.position
                    # Choose color: green for closest, yellow for 2nd, cyan for 3rd
                    if idx == 0:
                        color = (0, 255, 0)
                    elif idx == 1:
                        color = (255, 255, 0)
                    else:
                        color = (0, 255, 255)
                    pygame.draw.line(surface, color, (int(ship_x), int(ship_y)), (int(ast_x), int(ast_y)), 2)
                # For the closest one, also calculate angle diff
                targeting_bonus, angle_diff = self._calculate_targeting_info()
                angle_text = debug_font.render(f"Angle: {angle_diff:.1f}°", True, (0, 255, 0))
                target_text = debug_font.render(f"TargetBonus: {targeting_bonus:.2f}", True, (0, 255, 255))
                surface.blit(angle_text, (int(ship_x) + 20, int(ship_y) - 30))
                surface.blit(target_text, (int(ship_x) + 20, int(ship_y) - 60))
        # Draw HUD elements
        font = pygame.font.Font(None, 36)
        
        # Score
        score_text = font.render(f"SCORE: {self.score:06d}", True, (255, 255, 255))
        surface.blit(score_text, (10, 10))
        
        # Lives
        lives_text = font.render(f"LIVES: {max(0, self.lives)}", True, (255, 255, 255))
        surface.blit(lives_text, (10, 50))
        
        # Level
        level_text = font.render(f"LEVEL: {self.level}", True, (255, 255, 255))
        surface.blit(level_text, (10, 90))
        
        # Game over message
        if self.game_over:
            game_over_text = font.render("GAME OVER", True, (255, 0, 0))
            text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
            surface.blit(game_over_text, text_rect)
