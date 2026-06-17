import pygame
import sys
import math

# Dostosuj importy do nowej struktury
from src.core.drone import Drone
from src.config.config import SCREEN_WIDTH, SCREEN_HEIGHT
from src.core.flight_controller import FlightController

def test_manual_flight():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Fizyka Drona - TEST MANUALNY (Spacja by zmienić tryb)")
    clock = pygame.time.Clock()

    PPM = 200.0  
    drone = Drone((SCREEN_WIDTH / 2) / PPM, (SCREEN_HEIGHT / 2) / PPM)
    target_pos_m = ((SCREEN_WIDTH / 2) / PPM, ((SCREEN_HEIGHT / 2) / PPM) - 1.0)
    
    obstacles = [pygame.Rect(SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 - 200, 300, 50)]

    # Inicjalizacja kontrolera lotu do trybu z asystą
    controller = FlightController()

    mode = "CASCADE" # lub "RAW"
    font = pygame.font.SysFont("arial", 24)

    run = True
    while run:
        dt = min(clock.tick(60) / 1000.0, 0.05)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            # Spacja zmienia tryb
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                mode = "RAW" if mode == "CASCADE" else "CASCADE"
                # Resetujemy drona na środek ekranu po przełączeniu
                drone._x, drone._y = (SCREEN_WIDTH / 2) / PPM, (SCREEN_HEIGHT / 2) / PPM
                drone._angle = 0
                drone._vel_x, drone._vel_y, drone._angular_vel = 0, 0, 0

        keys = pygame.key.get_pressed()
        l_thrust, r_thrust = 0.0, 0.0

        if mode == "RAW":
            # --- BEZPOŚREDNIE STEROWANIE SILNIKAMI ---
            if keys[pygame.K_UP]:
                l_thrust, r_thrust = 1.0, 1.0
            if keys[pygame.K_LEFT]:
                l_thrust, r_thrust = 0.2, 0.8
            if keys[pygame.K_RIGHT]:
                l_thrust, r_thrust = 0.8, 0.2

        elif mode == "CASCADE":
            # --- STEROWANIE WIRTUALNYM JOYSTICKIEM (x, y) ---
            target_x = 0.0
            target_y = 0.0

            if keys[pygame.K_LEFT]:
                target_x = -1.0
            if keys[pygame.K_RIGHT]:
                target_x = 1.0
            if keys[pygame.K_UP]:
                target_y = -1.0
            if keys[pygame.K_DOWN]:
                target_y = 1.0

            l_thrust, r_thrust = controller.get_motor_thrusts(
                drone=drone,
                target_x=target_x,
                target_y=target_y
            )

        # 1. Przekazujemy ostateczny wynik do silników drona
        drone.set_engine_thrust(l_thrust, r_thrust)

        # 2. Fizyka
        drone.update(dt)
        drone.get_sensor_data(SCREEN_WIDTH, SCREEN_HEIGHT, obstacles, PPM)

        # --- RYSOWANIE ---
        screen.fill((20, 25, 30))
        for obs in obstacles:
            pygame.draw.rect(screen, (150, 50, 50), obs)
            pygame.draw.rect(screen, (255, 100, 100), obs, 2)

        drone.draw(screen, target_pos_m, PPM, show_sensors=True, show_thrust=True)

        # UI Overlay
        txt_info = font.render(f"Space to change mode between manual and cascade(flight controller)", True, (255, 255, 255))
        screen.blit(txt_info, (10, 10))

        color = (0, 255, 0) if mode == "CASCADE" else (255, 0, 0)
        txt_mode = font.render(f"MODE: {mode}", True, color)
        screen.blit(txt_mode, (10, 40))

        pygame.display.flip()

    pygame.quit()
    sys.exit()