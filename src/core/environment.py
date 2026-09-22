import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import pygame

from src.config.config import (
    GRID_SIZE_M,
    MAP_MARGIN_PX,
    PPM,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)
from src.core.map_generator import generate_grid_obstacles

# Difficulty tiers for scenario generation. Each tier specifies a distance band and the number of obstacles.
TIERS: dict[int, dict] = {
    1: {"dist_m": (0.5, 1.0), "obstacles": 0, "description": "target close, empty map"},
    2: {"dist_m": (1.0, 2.0), "obstacles": 0, "description": "medium distance, empty map"},
    3: {"dist_m": (2.0, 3.0), "obstacles": 1, "description": "one obstacle"},
    4: {"dist_m": (2.0, 4.0), "obstacles": 2, "description": "two obstacles"},
    5: {"dist_m": (2.0, 4.0), "obstacles": 3, "description": "three obstacles"},
    6: {"dist_m": (2.0, 4.0), "obstacles": 5, "description": "five obstacles"},
}

@dataclass(frozen=True)
class Scenario:
    """One instance of a scenario: start, target, and obstacles.

    Obstacles are stored as tuples. The object is sent
    to workers via pickle.
    """
    start_px: tuple[int, int]
    target_px: tuple[int, int]
    obstacles_px: tuple[tuple[int, int, int, int], ...]
    tier: int = 0

    def rects(self) -> list[pygame.Rect]:
        return [pygame.Rect(*o) for o in self.obstacles_px]

    def target_m(self, ppm: float) -> tuple[float, float]:
        return (self.target_px[0] / ppm, self.target_px[1] / ppm)

def generate_scenario_for_tier(tier: int) -> Scenario:
    spec = TIERS[tier]
    start_px, target_px = generate_start_and_target_in_band(*spec["dist_m"])
    obstacles = generate_grid_obstacles(
        SCREEN_WIDTH, SCREEN_HEIGHT, start_px, target_px,
        GRID_SIZE_M, spec["obstacles"], PPM
    )
    return Scenario(
        start_px=start_px,
        target_px=target_px,
        obstacles_px=tuple((r.x, r.y, r.width, r.height) for r in obstacles),
        tier=tier,
    )

def generate_start_and_target_in_band(
    min_dist_m: float, max_dist_m: float, max_tries: int = 500
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Generates pair start/target with distance in the given band (rejection sampling)."""
    for _ in range(max_tries):
        start = (random.randint(MAP_MARGIN_PX, SCREEN_WIDTH - MAP_MARGIN_PX),
                 random.randint(MAP_MARGIN_PX, SCREEN_HEIGHT - MAP_MARGIN_PX))
        target = (random.randint(MAP_MARGIN_PX, SCREEN_WIDTH - MAP_MARGIN_PX),
                  random.randint(MAP_MARGIN_PX, SCREEN_HEIGHT - MAP_MARGIN_PX))
        dist_m = math.hypot(target[0] - start[0], target[1] - start[1]) / PPM
        if min_dist_m <= dist_m <= max_dist_m:
            return start, target
    raise RuntimeError(
        f"could not generate a start/target pair in the band {min_dist_m}-{max_dist_m} m "
        f"after {max_tries} tries - check if the band fits within the map"
    )

def generate_start_and_target(width: int, height: int, margin: int, min_dist: float) -> tuple[tuple[int, int], tuple[int, int]]:
    """Generates a safe start and target point."""
    start_x = random.randint(margin, width - margin)
    start_y = random.randint(margin, height - margin)
    start_pos = (start_x, start_y)

    while True:
        target_x = random.randint(margin, width - margin)
        target_y = random.randint(margin, height - margin)
        target_pos = (target_x, target_y)

        dist = math.hypot(target_x - start_x, target_y - start_y)
        if dist >= min_dist * PPM:
            return start_pos, target_pos


def generate_scenarios(count: int, tier: int) -> list[Scenario]:
    """Generates K random scenarios in a given tier. The same set is given to all genomes in a generation."""
    return [generate_scenario_for_tier(tier) for _ in range(count)]

def load_holdout(path: str = "data/holdout.json") -> list[Scenario]:
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Scenario(
            start_px=tuple(d["start_px"]),
            target_px=tuple(d["target_px"]),
            obstacles_px=tuple(tuple(o) for o in d["obstacles_px"]),
            tier=int(d["tier"]),
        )
        for d in blob["scenarios"]
    ]