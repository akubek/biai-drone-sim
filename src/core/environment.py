import random
import pygame
import math
from src.config.config import SCREEN_WIDTH, SCREEN_HEIGHT, PPM

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