import pygame
import random
from collections import deque

def _get_grid_coords(px_pos: tuple[int, int], grid_size: int) -> tuple[int, int]:
    """Zwraca indeks (kolumna, wiersz) dla danej pozycji w pikselach."""
    return px_pos[0] // grid_size, px_pos[1] // grid_size

def _is_solvable(grid: list[list[bool]], start_idx: tuple[int, int], target_idx: tuple[int, int], cols: int, rows: int) -> bool:
    """Sprawdza za pomocą algorytmu BFS, czy istnieje ścieżka od startu do celu."""
    queue = deque([start_idx])
    visited = set([start_idx])

    # Możliwe ruchy: góra, dół, lewo, prawo
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while queue:
        current = queue.popleft()
        
        if current == target_idx:
            return True # Znaleziono drogę!

        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            
            # Sprawdzamy czy nie wychodzimy poza mapę i czy nie uderzamy w ścianę
            if 0 <= nx < cols and 0 <= ny < rows:
                if not grid[nx][ny] and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
                    
    return False # Brak przejścia

def generate_grid_obstacles(
    width_px: int, 
    height_px: int, 
    start_px: tuple[int, int], 
    target_px: tuple[int, int], 
    grid_size_m: float, 
    max_obstacles: int, 
    safe_zone: int,
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
    
    # 2. Tworzymy listę wszystkich dozwolonych komórek (poza safe zone)
    available_cells = []
    for x in range(cols):
        for y in range(rows):
            dist_to_start = max(abs(x - start_idx[0]), abs(y - start_idx[1]))
            dist_to_target = max(abs(x - target_idx[0]), abs(y - target_idx[1]))
            
            if dist_to_start > safe_zone and dist_to_target > safe_zone:
                available_cells.append((x, y))

    # 3. Przemieszanie indeksów (losowość mapy)
    random.shuffle(available_cells)

    # 4. Konstruktywne dodawanie przeszkód z walidacją BFS
    obstacles_placed = 0
    
    for x, y in available_cells:
        print(f"Próba postawienia przeszkody na ({x}, {y})... \n")
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
                print(f"Przeszkoda na ({x}, {y}) została zatwierdzona. \n")
                rect = pygame.Rect(x * grid_size_px, y * grid_size_px, grid_size_px, grid_size_px)
                obstacles.append(rect)

    return obstacles