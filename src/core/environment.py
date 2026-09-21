import math
import random
from dataclasses import dataclass

import pygame

from src.config.config import (
    GRID_SIZE_M,
    MAP_MARGIN_PX,
    MIN_SPAWN_DIST_M,
    PPM,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)
from src.core.map_generator import generate_grid_obstacles


def generate_start_and_target(width: int, height: int, margin: int, min_dist: float) -> tuple[tuple[int, int], tuple[int, int]]:
    """Generuje bezpieczny punkt startowy i docelowy."""
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

@dataclass(frozen=True)
class Scenario:
    """One instance of a scenario: start, target, and obstacles.

    Obstacles are stored as tuples. The object is sent
    to workers via pickle.
    """
    start_px: tuple[int, int]
    target_px: tuple[int, int]
    obstacles_px: tuple[tuple[int, int, int, int], ...]

    def rects(self) -> list[pygame.Rect]:
        return [pygame.Rect(*o) for o in self.obstacles_px]

    def target_m(self, ppm: float) -> tuple[float, float]:
        return (self.target_px[0] / ppm, self.target_px[1] / ppm)


def generate_scenarios(count: int, num_obstacles: int) -> list[Scenario]:
    """Generates K random scenarios. The same set is given to all genomes in a generation."""
    scenarios = []
    for _ in range(count):
        start_px, target_px = generate_start_and_target(
            SCREEN_WIDTH, SCREEN_HEIGHT, MAP_MARGIN_PX, MIN_SPAWN_DIST_M
        )
        obstacles = generate_grid_obstacles(
            SCREEN_WIDTH, SCREEN_HEIGHT, start_px, target_px,
            GRID_SIZE_M, num_obstacles, PPM
        )
        scenarios.append(Scenario(
            start_px=start_px,
            target_px=target_px,
            obstacles_px=tuple((r.x, r.y, r.width, r.height) for r in obstacles),
        ))
    return scenarios