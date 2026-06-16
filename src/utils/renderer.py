from src.core.constants import TARGET_SIZE_PX


import pygame
from src.core.drone import Drone

def render_simulation(
    screen: pygame.Surface,
    drones: list[Drone],
    target_pos_px: tuple[int, int],
    obstacles: list[pygame.Rect],
    PPM: float,
) -> None:
    """Rysuje całą klatkę symulacji."""
    screen.fill((20, 25, 30))

    for obs in obstacles:
        pygame.draw.rect(screen, (150, 50, 50), obs)
        pygame.draw.rect(screen, (255, 100, 100), obs, 2)

    pygame.draw.circle(screen, (0, 255, 0), target_pos_px, TARGET_SIZE_PX, 2)
    pygame.draw.circle(screen, (0, 255, 0), target_pos_px, 3)

    target_pos_m = (target_pos_px[0] / PPM, target_pos_px[1] / PPM)

    for i, drone in enumerate(drones):
        is_champion = i == 0
        drone.draw(
            screen=screen,
            target_pos_m=target_pos_m,
            PPM=PPM,
            show_sensors=is_champion,
            show_thrust=True,
            show_hitbox=False,
        )

    pygame.display.flip()