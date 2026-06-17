from src.config.config import TARGET_SIZE_PX


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

def render_neat_hud(
    screen: pygame.Surface, 
    font: pygame.font.Font, 
    generation: int, 
    alive_count: int, 
    pop_size: int, 
    best_fitness: float,
    current_time_sec: float
) -> None:
    """Rysuje interfejs (HUD) widoczny tylko podczas treningu NEAT."""
    
    # --- STATYSTYKI TRENINGU ---
    stats_texts = [
        f"Generation: {generation}",
        f"Alive drones: {alive_count} / {pop_size}",
        f"Best fitness: {best_fitness:.1f}",
        f"Trial time: {current_time_sec:.1f} s"
    ]
    
    # Rysowanie statystyk w lewym górnym rogu
    for i, text in enumerate(stats_texts):
        surface = font.render(text, True, (255, 255, 255))
        screen.blit(surface, (10, 10 + (i * 30)))

    # --- KLAWISZOLOGIA (Prawy górny róg) ---
    controls_texts = [
        "[R] - Disable/Enable rendering",
        "[U] - Enable/Disable uncapped frames",
        "[1] - Slow down frames",
        "[2] - Normal speed",
    ]
    
    screen_width = screen.get_width()
    for i, text in enumerate(controls_texts):
        surface = font.render(text, True, (200, 200, 200))  # Lekko szary dla kontrastu
        text_rect = surface.get_rect(topright=(screen_width - 10, 10 + (i * 30)))
        screen.blit(surface, text_rect)