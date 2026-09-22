import math
import random
from collections import deque

import pygame

from src.config.config import OBSTACLE_CORRIDOR_M


def _get_grid_coords(px_pos: tuple[int, int], grid_size: int) -> tuple[int, int]:
    """Zwraca indeks (kolumna, wiersz) dla danej pozycji w pikselach."""
    return px_pos[0] // grid_size, px_pos[1] // grid_size

def _is_solvable(grid: list[list[bool]], start_idx: tuple[int, int], target_idx: tuple[int, int], cols: int, rows: int) -> bool:
    """Sprawdza za pomocą algorytmu BFS, czy istnieje ścieżka od startu do celu."""
    queue = deque([start_idx])
    visited = {start_idx}

    # Możliwe ruchy: góra, dół, lewo, prawo
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while queue:
        current = queue.popleft()
        
        if current == target_idx:
            return True # Znaleziono drogę!

        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            
            # Sprawdzamy czy nie wychodzimy poza mapę i czy nie uderzamy w ścianę
            if 0 <= nx < cols and 0 <= ny < rows and not grid[nx][ny] and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))
                    
    return False # Brak przejścia

def _dist_point_to_segment_m(px: float, py: float,
                             ax: float, ay: float,
                             bx: float, by: float) -> float:
    """Odleglosc punktu od odcinka - z obcieciem rzutu do [0, 1]."""
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby
    if denom == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / denom))
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))

def generate_grid_obstacles(
    width_px: int, 
    height_px: int, 
    start_px: tuple[int, int], 
    target_px: tuple[int, int], 
    grid_size_m: float, 
    max_obstacles: int,
    PPM: float
) -> list[pygame.Rect]:
    """Konstruktywny generator mapy, gwarantujący przejezdność i stałą liczbę przeszkód."""
    
    grid_size_px = int(grid_size_m * PPM)
    cols = width_px // grid_size_px
    rows = height_px // grid_size_px
    
    start_idx = _get_grid_coords(start_px, grid_size_px)
    target_idx = _get_grid_coords(target_px, grid_size_px)

    # 1. Inicjalizacja pustej siatki
    grid = [[False for _ in range(rows)] for _ in range(cols)]
    
    safe_zone_m: float = grid_size_m / 2.0 + 0.2 # half of grid cell + hald of drone size (drone size = 0.35 + margin)

    start_x_m, start_y_m = start_px[0] / PPM, start_px[1] / PPM
    target_x_m, target_y_m = target_px[0] / PPM, target_px[1] / PPM

    near: list[tuple[int, int]] = []
    far: list[tuple[int, int]] = []
    for x in range(cols):
        for y in range(rows):
            cell_center_x_m = (x * grid_size_px + grid_size_px / 2) / PPM
            cell_center_y_m = (y * grid_size_px + grid_size_px / 2) / PPM

            dist_to_start_m = math.hypot(cell_center_x_m - start_x_m,
                                         cell_center_y_m - start_y_m)
            dist_to_target_m = math.hypot(cell_center_x_m - target_x_m,
                                          cell_center_y_m - target_y_m)
            if dist_to_start_m <= safe_zone_m or dist_to_target_m <= safe_zone_m:
                continue

            d_line_m = _dist_point_to_segment_m(
                cell_center_x_m, cell_center_y_m,
                start_x_m, start_y_m, target_x_m, target_y_m,
            )
            (near if d_line_m <= OBSTACLE_CORRIDOR_M else far).append((x, y))

    random.shuffle(near)
    random.shuffle(far)
    available_cells = near + far

    # 4. Konstruktywne dodawanie przeszkód z walidacją BFS
    obstacles_placed = 0
    
    for x, y in available_cells:
        if obstacles_placed >= max_obstacles:
            break  # Osiągnęliśmy cel!
            
        # Próbujemy postawić ścianę
        grid[x][y] = True
        
        # Sprawdzamy, czy mapa wciąż jest przejezdna
        if _is_solvable(grid, start_idx, target_idx, cols, rows):
            obstacles_placed += 1  # Super, zostawiamy i liczymy
        else:
            grid[x][y] = False     # Błąd! Ta ściana odcinała drogę, cofamy!

    # 5. Konwersja siatki na obiekty pygame.Rect
    obstacles = []
    for x in range(cols):
        for y in range(rows):
            if grid[x][y]:
                rect = pygame.Rect(x * grid_size_px, y * grid_size_px, grid_size_px, grid_size_px)
                obstacles.append(rect)

    if obstacles_placed < max_obstacles:
        print(f"WARNING: ordered {max_obstacles} obstacles, placed {obstacles_placed} "
              f"(no space left on grid {cols}x{rows})")

    return obstacles