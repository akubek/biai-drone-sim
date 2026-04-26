import heapq
import math
import pygame


def get_expert_path(
    start_px,
    target_px,
    obstacles,
    drone_radius_px,
    grid_size=20,
    screen_w=800,
    screen_h=600,
):
    """
    Znajduje optymalną ścieżkę używając algorytmu A*.
    Zwraca listę punktów (x, y) lub pustą listę, jeśli brak ścieżki.
    """
    # 1. "Pompowanie" przeszkód (Configuration Space)
    # Powiększamy przeszkodę o średnicę drona + mały margines bezpieczeństwa
    margin = (drone_radius_px * 2) + 10
    inflated_obs = [obs.inflate(margin, margin) for obs in obstacles]

    # 2. Konwersja pikseli na koordynaty siatki (Grid)
    start_node = (int(start_px[0] // grid_size), int(start_px[1] // grid_size))
    target_node = (int(target_px[0] // grid_size), int(target_px[1] // grid_size))

    # 3. Inicjalizacja algorytmu A*
    open_set = []
    heapq.heappush(open_set, (0, start_node))
    came_from = {}
    g_score = {start_node: 0.0}

    # Funkcja heurystyki (Odległość w linii prostej)
    def heuristic(node):
        return math.hypot(target_node[0] - node[0], target_node[1] - node[1])

    # Funkcja sprawdzająca kolizje
    def is_collision(nx, ny):
        # Sprawdzenie granic ekranu
        px, py = nx * grid_size, ny * grid_size
        if px < 0 or py < 0 or px >= screen_w or py >= screen_h:
            return True
        # Sprawdzenie powiększonych przeszkód
        test_rect = pygame.Rect(px, py, grid_size, grid_size)
        return test_rect.collidelist(inflated_obs) != -1

    # 4. Pętla główna A*
    while open_set:
        _, current = heapq.heappop(open_set)

        if current == target_node:
            # Sukces! Odtwarzamy ścieżkę
            path = []
            while current in came_from:
                # Zamieniamy z powrotem grid na piksele (środek komórki)
                path.append(
                    (
                        current[0] * grid_size + grid_size // 2,
                        current[1] * grid_size + grid_size // 2,
                    )
                )
                current = came_from[current]
            path.reverse()
            return path

        # Sprawdzanie 8 sąsiadów (Ruchy pion/poziom/skosy)
        directions = [
            (0, 1),
            (1, 0),
            (0, -1),
            (-1, 0),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)

            if is_collision(neighbor[0], neighbor[1]):
                continue

            # Koszt: 1.0 dla prostej, 1.414 (pierwiastek z 2) dla skosów
            cost = 1.414 if dx != 0 and dy != 0 else 1.0
            tentative_g = g_score[current] + cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor)
                heapq.heappush(open_set, (f_score, neighbor))

    return []  # Brak możliwej ścieżki (zablokowana mapa)
