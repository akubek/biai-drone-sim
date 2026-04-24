import math
import pygame


def draw_vector(
    screen: pygame.Surface,
    start_pos: tuple[float, float],
    angle_deg: float,
    length: float,
    color: tuple[int, int, int] = (255, 255, 0),
) -> None:
    if abs(length) < 1:
        return  # Nie rysuj mikroskopijnych wektorów

    # Oblicz koniec linii (wektor siły)
    rad: float = math.radians(angle_deg - 90)
    end_x: float = start_pos[0] + math.cos(rad) * length
    end_y: float = start_pos[1] + math.sin(rad) * length

    # Rysuj główną linię
    _ = pygame.draw.line(screen, color, start_pos, (end_x, end_y), 2)

    # Rysuj grot strzałki (mały trójkąt na końcu)
    arrow_size = 5
    angle_arrow: float = rad + math.pi  # Odwracamy o 180 stopni

    p1: tuple[float, float] = (
        end_x + math.cos(angle_arrow + 0.5) * arrow_size,
        end_y + math.sin(angle_arrow + 0.5) * arrow_size,
    )
    p2: tuple[float, float] = (
        end_x + math.cos(angle_arrow - 0.5) * arrow_size,
        end_y + math.sin(angle_arrow - 0.5) * arrow_size,
    )

    _ = pygame.draw.polygon(screen, color, [(end_x, end_y), p1, p2])
