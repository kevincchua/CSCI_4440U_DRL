# code/games/asteroids.py
from __future__ import annotations
import random
import math
from typing import Callable, Tuple
import numpy as np

# ---- Score tracking utility -------------------------------------------

class _ScoreTracker:
    """
    Tracks previous score and exposes score increase detection.
    Used by reward wrapper to detect when agent scores points.
    """
    def __init__(self):
        """Initialize with score of 0."""
        self.prev = 0

    def step(self, info: dict) -> Tuple[int, bool]:
        """
        Update score tracking and detect if score increased.
        
        Args:
            info: Info dict containing current score
        
        Returns:
            tuple[int, bool]: (current_score, score_increased_this_step)
        """
        score = int(info.get("score", 0))
        increased = score > self.prev
        self.prev = score
        return score, increased

    def reset(self):
        """Reset score tracking (call at episode end)."""
        self.prev = 0

def _wrap_with_tracker(core_fn) -> Callable:
    """
    Decorator that wraps reward functions with score tracking.
    Adapts reward functions to the GameEnv signature and handles
    episode-local score tracking with automatic reset.
    
    All distance and targeting calculations are now done in asteroids_core.py
    and passed through the info dict for efficiency.
    
    Args:
        core_fn: Core reward function with signature (score_inc, terminated, info, score)
    
    Returns:
        Callable: Wrapped reward function with GameEnv signature
    """
    tracker = _ScoreTracker()

    def reward(obs, base, terminated: bool, info: dict) -> float:
        """
        Wrapped reward function that handles score tracking.
        
        Args:
            obs: Observation (unused - all info comes from info dict now)
            base: Base reward from environment (unused)
            terminated: Whether episode ended
            info: Info dict with all calculated values from core
        
        Returns:
            float: Calculated reward value
        """
        _score, inc = tracker.step(info)
        r = float(core_fn(inc, terminated, info, _score))
        
        # Reset tracker at episode end
        if terminated or info.get("episode_end", False):
            tracker.reset()
        
        return r

    return reward

# ---- Distance utility functions ---------------------------------------

def distance_band_bonus_single(distance, ideal_min=100, ideal_max=300, penalty_scale=0.01):
    """
    Calculate reward for staying within an ideal distance band from nearest asteroid.
    Provides positive reward within the band, linear penalty outside.
    
    Args:
        distance: Distance to nearest asteroid
        ideal_min: Minimum ideal distance (closer = penalty)
        ideal_max: Maximum ideal distance (farther = penalty)
        penalty_scale: Scale factor for penalty (higher = steeper penalty)
    
    Returns:
        float: Reward value (+1.0 in band, negative outside)
    """
    if ideal_min <= distance <= ideal_max:
        return 1.0  # In sweet spot
    elif distance < ideal_min:
        return -penalty_scale * (ideal_min - distance)  # Too close penalty
    else:
        return -penalty_scale * (distance - ideal_max)  # Too far penalty

def distance_band_bonus_multi(closest_distances, ideal_min=100, ideal_max=300, penalty_scale=0.01):
    """
    Calculate reward for staying within ideal distance from multiple asteroids.
    Uses average distance to closest N asteroids for more stable positioning.
    
    Args:
        closest_distances: List of distances to closest N asteroids
        ideal_min: Minimum ideal distance
        ideal_max: Maximum ideal distance  
        penalty_scale: Scale factor for penalty
    
    Returns:
        float: Reward value based on average distance to multiple asteroids
    """

    return sum(
        distance_band_bonus_single(d, ideal_min, ideal_max, penalty_scale)
        for d in closest_distances
    )/len(closest_distances)
    avg_distance = sum(closest_distances) / len(closest_distances)
    return distance_band_bonus_single(avg_distance, ideal_min, ideal_max, penalty_scale)

# ---- Three distinct asteroids reward personas ------------------------

@_wrap_with_tracker
def survivor(score_inc: bool, terminated: bool, info: dict, score: int) -> float:
    """
    SURVIVOR PERSONA: Defensive gameplay focused on staying alive.

    Reward structure:
    - +0.1 per frame (survival bonus)
    - +distance band bonus (prefers safer distances 150-400)
    - +1.0 when score increases  
    - -1.0 for hyperspace usage (unless within 10 units of closest asteroid, then +10)
    - -5.0 for collisions
    - -10.0 on death
    - NO targeting bonus (not focused on offense)
    """
    if terminated:
        return -10.0  # Heavy death penalty
    r = 0.1  # Base survival bonus per frame
    # Distance bonus using closest 3 asteroids
    closest_3 = info.get("distances_to_closest_3", [800.0, 800.0, 800.0])
    r += distance_band_bonus_multi(closest_3, 150, 400, 0.005)
    # Score increase bonus
    r += info.get("score_delta", 0) * 0.2
    r += info.get("asteroids_destroyed", 0) * 2.0
    # Hyperspace logic
    if info.get("hyperspace_used", False):
        nearest = min(closest_3) if closest_3 else 800.0
        if nearest <= 10.0:
            r += 10.0  # Smart escape gets a big bonus
    # Heavy collision penalty
    if info.get("collision", False):
        r -= 5.0
    return r


@_wrap_with_tracker  
def hunter(score_inc: bool, terminated: bool, info: dict, score: int) -> float:
    """
    HUNTER PERSONA: Aggressive gameplay focused on accurate shooting.
    
    Reward structure:
    - +targeting²  * bullets_fired when well-aimed (targeting > 2.5)
    - -0.5 * bullets_fired when poorly aimed (targeting < 0.5) 
    - -0.1 * bullets_fired (general anti-spam penalty)
    - +5.0 per asteroid destroyed
    - +50.0 for level completion
    - +0.2 * score_delta (point progression)
    - +distance band bonus (stay in tactical range 100-300)
    - -0.2 when moving too slowly (anti-camping)
    - -50.0 on death
    
    This persona encourages active, accurate hunting with good positioning.
    
    Args:
        score_inc: Whether score increased this step
        terminated: Whether episode ended
        info: Info dict with all game state information  
        score: Current total score
    
    Returns:
        float: Calculated reward value
    """
    if terminated:
        return -50.0
    r = 0.01

    fired = info.get("bullets_fired", 0)
    ship_speed = info.get("ship_speed", 0.0)
    targeting = info.get("targeting_bonus", 0.0)

    if fired > 0:
        if targeting > 3.0:
            r += (targeting ** 2) * fired
        elif targeting > 1.0:
            r += targeting * fired
        r -= fired * 0.1
    else:
        # New: Add delta_targeting_bonus when no bullet is fired
        r += info.get("targeting_bonus_delta", 0.0)
    r += info.get("asteroids_destroyed", 0) * 5.0
    if info.get("level_completed", False):
        r += 50.0
    r += info.get("score_delta", 0) * 0.2
    #closest_3 = info.get("distances_to_closest_3", [800.0, 800.0, 800.0])
    #r += distance_band_bonus_multi(closest_3, 100, 300, 0.01)
    if ship_speed < 0.02:
        r -= 0.2

    return r


@_wrap_with_tracker
def speedrunner(score_inc: bool, terminated: bool, info: dict, score: int) -> float:
    """
    SPEEDRUNNER PERSONA: Fast-paced gameplay focused on level completion.
    
    Reward structure:
    - -0.001 per frame (time pressure)
    - +25.0 for level completion (main objective)
    - +2.0 per asteroid destroyed (progress toward level completion)
    - +0.2 * ship_speed (movement bonus)
    - +0.02 * score_delta (point momentum) 
    - +0.3 * targeting_bonus (efficient aiming)
    - +distance band bonus (moderate engagement range 80-250)
    - -3.0 on death (moderate penalty - speed over safety)
    
    This persona encourages fast, efficient play with emphasis on level completion.
    
    Args:
        score_inc: Whether score increased this step
        terminated: Whether episode ended
        info: Info dict with all game state information
        score: Current total score
    
    Returns:
        float: Calculated reward value
    """
    # Death penalty (moderate - speed is prioritized over safety)
    if terminated:
        return -3.0
    r = -0.001  # Time pressure - every frame costs a small amount
    
    fired = info.get("bullets_fired", 0)
    targeting = info.get("targeting_bonus", 0.0)
    ship_speed = info.get("ship_speed", 0.0) 
    # Level completion (main goal for speedrunner)
    if info.get("level_completed", False):
        r += 25.0
    
    # Asteroid destruction (progress toward level completion)
    r += info.get("asteroids_destroyed", 0) * 2.0
    
    # Movement bonus (encourage active play)
    r += ship_speed * 0.2
    
    # Score momentum (maintain forward progress)
    r += info.get("score_delta", 0) * 0.02
    
    # Targeting bonus (efficient aiming saves time)
    r += info.get("targeting_bonus_delta", 0.0)
    
    if fired > 0:
        r += targeting
    # Distance bonus (moderate engagement distance - not too far from action)
    distance = info.get("distance_to_nearest", 800.0)
    r += distance_band_bonus_single(distance, 80, 250, 0.008)
    
    return r

# ---- Reference reward functions -----------------------------------

@_wrap_with_tracker
def baseline(score_inc: bool, terminated: bool, info: dict, score: int) -> float:
    """
    BASELINE PERSONA: Random rewards for benchmarking.
    Used to test that agents are actually learning vs random performance.
    
    Returns:
        float: Random reward between -0.5 and 0.5
    """
    return random.random() - 0.5

@_wrap_with_tracker
def simple(score_inc: bool, terminated: bool, info: dict, score: int) -> float:
    """
    SIMPLE PERSONA: Basic reward structure for testing.
    Simple reward that just encourages basic survival and asteroid destruction.
    
    Reward structure:
    - +0.1 per frame (survival)
    - +1.0 per asteroid destroyed
    
    Returns:
        float: Simple calculated reward
    """
    r = 0.1  # Survival bonus
    r += info.get("asteroids_destroyed", 0) * 1.0  # Destruction bonus
    return r