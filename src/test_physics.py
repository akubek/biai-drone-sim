import pygame
import sys
from src.drone import Drone
from src.constants import SCREEN_WIDTH, SCREEN_HEIGHT


def test_manual_flight():
    """Manualny test fizyki drona za pomocą klawiatury."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Fizyka Drona - TEST MANUALNY")
    clock = pygame.time.Clock()

    PPM = 200.0  # Nasza skala: 200 pikseli to 1 metr

    # Środek ekranu w metrach
    start_x_m = (SCREEN_WIDTH / 2) / PPM
    start_y_m = (SCREEN_HEIGHT / 2) / PPM

    drone = Drone(start_x_m, start_y_m)

    # Cel w metrach (np. lekko nad dronem)
    target_pos_m = (start_x_m, start_y_m - 1.0)

    # Przykładowa przeszkoda w pikselach (dla testu promieni i hitboxa)
    obstacles = [
        pygame.Rect(SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 - 200, 300, 50)
    ]

    run = True
    while run:
        # clock.tick zwraca czas w milisekundach, dzielimy przez 1000, by mieć sekundy (dt)
        # min() zabezpiecza przed wielkim "skokiem" fizyki przy lagu okna
        dt = min(clock.tick(60) / 1000.0, 0.05)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # --- STEROWANIE KLAWIATURĄ ---
        keys = pygame.key.get_pressed()
        l_thrust = 0.0
        r_thrust = 0.0

        # Strzałka w górę - pełna moc obu silników (100%)
        if keys[pygame.K_UP]:
            l_thrust = 1.0
            r_thrust = 1.0
        # Strzałka w lewo - prawy silnik daje mocniej, dron przechyla się w lewo
        if keys[pygame.K_LEFT]:
            l_thrust = 0.2
            r_thrust = 0.8
        # Strzałka w prawo - lewy silnik daje mocniej, dron przechyla się w prawo
        if keys[pygame.K_RIGHT]:
            l_thrust = 0.8
            r_thrust = 0.2

        # 1. Przekazujemy sygnał do drona
        drone.set_engine_thrust(l_thrust, r_thrust)

        # 2. Dron liczy swoją fizykę
        drone.update(dt)

        # 3. Dron sam sprawdza kolizje (tylko wypisujemy w konsoli dla testu)
        is_hit = drone.check_collision(SCREEN_WIDTH, SCREEN_HEIGHT, obstacles, PPM)
        if is_hit:
            print("💥 BUM! Kolizja!")

        # 4. Aktualizujemy sensory (aby widzieć linie na ekranie)
        drone.get_sensor_data(SCREEN_WIDTH, SCREEN_HEIGHT, obstacles, PPM)

        # --- RYSOWANIE ---
        screen.fill((20, 25, 30))

        # Rysowanie przeszkody
        for obs in obstacles:
            pygame.draw.rect(screen, (150, 50, 50), obs)
            pygame.draw.rect(screen, (255, 100, 100), obs, 2)

        # Rysowanie celu
        px_target = (int(target_pos_m[0] * PPM), int(target_pos_m[1] * PPM))
        pygame.draw.circle(screen, (0, 255, 0), px_target, 10, 2)

        # Rysowanie drona ze wszystkimi flagami debugowania na True
        drone.draw(
            screen,
            target_pos_m,
            PPM,
            show_radar=True,
            show_sensors=True,
            show_thrust=True,
            show_hitbox=True,
        )

        pygame.display.flip()

    pygame.quit()
    sys.exit()
