import random
import pygame
import math
from .constants import SCREEN_WIDTH, SCREEN_HEIGHT, PPM

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

def generate_obstacles(start_pos: tuple[int, int], target_pos: tuple[int, int], num_obstacles: int = 5) -> list[pygame.Rect]:
    """Generuje losowe przeszkody, nie blokując startu ani celu."""
    obstacles = []
    start_rect = pygame.Rect(start_pos[0] - 50, start_pos[1] - 50, 100, 100)
    target_rect = pygame.Rect(target_pos[0] - 50, target_pos[1] - 50, 100, 100)

    for _ in range(num_obstacles):
        w, h = random.randint(50, 150), random.randint(50, 150)
        x, y = random.randint(0, SCREEN_WIDTH - w), random.randint(0, SCREEN_HEIGHT - h)
        new_rect = pygame.Rect(x, y, w, h)

        if not new_rect.colliderect(start_rect) and not new_rect.colliderect(target_rect):
            obstacles.append(new_rect)

    return obstacles