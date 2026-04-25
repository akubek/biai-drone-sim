import neat
import pygame
import math
import sys
import random
import pickle

from pathlib import Path
from .drone import Drone
from .constants import *

pygame.font.init()
STAT_FONT = pygame.font.SysFont("arial", 50)

show_simulation = True
EVOLUTION_CYCLES = 250
SIMULATION_TIME = 10
TARGET_SIZE = 50
BASE_CRASH_PENALTY = 20.0
stagnation_limit = FPS * 2.5

# =====================================================================
# METODY POMOCNICZE (ŚRODOWISKO I GRAFIKA)
# =====================================================================


def generate_obstacles(
    target_pos: tuple[int, int], num_obstacles: int = 5
) -> list[pygame.Rect]:
    """Generuje losowe przeszkody, upewniając się, że nie blokują startu ani celu."""
    obstacles = []
    start_rect = pygame.Rect(SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT // 2 - 50, 100, 100)
    target_rect = pygame.Rect(target_pos[0] - 50, target_pos[1] - 50, 100, 100)

    for _ in range(num_obstacles):
        w, h = random.randint(50, 150), random.randint(50, 150)
        x, y = random.randint(0, SCREEN_WIDTH - w), random.randint(0, SCREEN_HEIGHT - h)
        new_rect = pygame.Rect(x, y, w, h)

        if not new_rect.colliderect(start_rect) and not new_rect.colliderect(
            target_rect
        ):
            obstacles.append(new_rect)

    return obstacles


def check_collisions(
    drone: Drone, obstacles: list[pygame.Rect], drone_radius: int = 20
) -> tuple[bool, bool]:
    """Sprawdza kolizje drona ze ścianami i przeszkodami."""
    hit_wall = (
        drone.x - drone_radius < 0
        or drone.x + drone_radius > SCREEN_WIDTH
        or drone.y - drone_radius < 0
        or drone.y + drone_radius > SCREEN_HEIGHT
    )
    drone_rect = pygame.Rect(
        drone.x - drone_radius,
        drone.y - drone_radius,
        drone_radius * 2,
        drone_radius * 2,
    )
    hit_obstacle = any(obs.colliderect(drone_rect) for obs in obstacles)

    return hit_wall, hit_obstacle


def render_simulation(
    screen: pygame.Surface,
    drones: list[Drone],
    target_pos: tuple[int, int],
    obstacles: list[pygame.Rect],
) -> None:
    """Rysuje całą klatkę symulacji."""
    screen.fill((20, 25, 30))

    # Rysowanie przeszkód
    for obs in obstacles:
        pygame.draw.rect(screen, (150, 50, 50), obs)
        pygame.draw.rect(screen, (255, 100, 100), obs, 2)

    # Rysowanie celu
    pygame.draw.circle(screen, (0, 255, 0), target_pos, TARGET_SIZE, 2)
    pygame.draw.circle(screen, (0, 255, 0), target_pos, 3)

    # Rysowanie dronów
    for drone in drones:
        drone.draw(
            screen,
            target_pos,
            getattr(drone, "l_thrust", 0.5),
            getattr(drone, "r_thrust", 0.5),
        )
    pygame.display.flip()


def remove_drone(
    index: int,
    drones: list[Drone],
    nets: list[neat.nn.FeedForwardNetwork],
    ge: list[neat.DefaultGenome],
) -> None:
    drones.pop(index)
    nets.pop(index)
    ge.pop(index)


# =====================================================================
# GŁÓWNA LOGIKA EWOLUCJI
# =====================================================================
def is_target_visible(start_pos, target_pos, obstacles):
    """Sprawdza, czy linia między startem a celem jest przecięta przez przeszkodę."""
    line = (start_pos, target_pos)
    for obs in obstacles:
        if obs.clipline(line):
            return False
    return True


def eval_genomes(genomes, config) -> None:
    global show_simulation
    screen = pygame.display.get_surface()
    clock = pygame.time.Clock()

    nets, drones, ge = [], [], []

    # 1. Środowisko
    target_pos = (
        random.randint(100, SCREEN_WIDTH - 100),
        random.randint(100, SCREEN_HEIGHT - 100),
    )
    obstacles = generate_obstacles(target_pos)

    # --- OBLICZANIE TRUDNOŚCI GENERACJI ---
    start_pos = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    visible = is_target_visible(start_pos, target_pos, obstacles)

    # Dystans startowy (im dalej, tym trudniej)
    dist_start = math.hypot(target_pos[0] - start_pos[0], target_pos[1] - start_pos[1])
    dist_factor = dist_start / (SCREEN_WIDTH / 2)  # Normalizacja względem połowy ekranu

    # Finalny mnożnik: bonus za brak widoczności (1.5x) i za dystans
    difficulty_multiplier = (1.5 if not visible else 1.0) * (1.0 + dist_factor)

    # 2. Populacja
    for genome_id, genome in genomes:
        genome.fitness = 100.0  # STARTOWY KAPITAŁ
        nets.append(neat.nn.FeedForwardNetwork.create(genome, config))

        new_drone = Drone(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        new_drone.hover_frames = 0
        new_drone.idle_frames = 0
        new_drone.initial_dist = dist_start
        new_drone.min_dist = new_drone.initial_dist
        new_drone.has_touched_target = False  # Flaga dla bonusu

        drones.append(new_drone)
        ge.append(genome)

    run = True
    frames = 0
    max_frames = FPS * SIMULATION_TIME

    # 3. Pętla Życia
    while run and len(drones) > 0:
        frames += 1
        if frames > max_frames:
            break

        if show_simulation or frames % 10 == 0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_h:
                    show_simulation = not show_simulation

        drones_to_remove = []

        for i, drone in enumerate(drones):
            inputs = drone.get_inputs(
                target_pos[0], target_pos[1], SCREEN_WIDTH, SCREEN_HEIGHT, obstacles
            )
            output = nets[i].activate(inputs)

            drone.l_thrust, drone.r_thrust = output[0], output[1]
            drone.apply_thrust(drone.l_thrust, drone.r_thrust)
            drone.update()

            # ==========================================
            # SMOOTHNESS BONUS (Premia za płynność)
            # ==========================================
            # Liczymy o ile zmieniła się moc silników od ostatniej klatki
            delta_l = abs(drone.l_thrust - drone.prev_l_thrust)
            delta_r = abs(drone.r_thrust - drone.prev_r_thrust)

            # Sumaryczna "szorstkość" (jitter). Max to 2.0 (jeśli oba skoczyły z 0 na 1)
            jitter = delta_l + delta_r

            # Zamieniamy to na bonus (płynność).
            # 1.0 oznacza idealnie stały ciąg, 0.0 oznacza maksymalne szarpanie.
            smoothness = max(0.0, 1.0 - jitter)

            # Dodajemy mały, stały bonus za każdą klatkę płynnego lotu.
            # To sprawi, że drony "drgające" będą miały wyraźnie mniej punktów
            # od tych "płynnych" przy tym samym dystansie do celu.
            ge[i].fitness += smoothness * 0.5 * difficulty_multiplier

            dist = math.hypot(drone.x - target_pos[0], drone.y - target_pos[1])
            time_progress = frames / max_frames
            time_decay = 1.0 - (0.8 * time_progress)

            # Jeśli dron jest w miarę blisko celu, nagradzaj brak niepotrzebnych obrotów
            if dist < TARGET_SIZE * 2:
                # angular_vel to zmiana kąta w czasie.
                # Chcemy, żeby była bliska 0 przy precyzyjnym manewrowaniu.
                stability_bonus = max(0.0, 1.0 - abs(drone.angular_vel / 5.0))
                ge[i].fitness += stability_bonus * 0.2 * difficulty_multiplier

            # --- NAGRODA: Discovery Bonus (Tylko raz!) ---
            if not drone.has_touched_target and dist < TARGET_SIZE:
                # Duży bonus za samo dotarcie, skalowany trudnością
                ge[i].fitness += 500.0 * difficulty_multiplier * time_decay
                drone.has_touched_target = True

            # --- FITNESS: Lenistwo ---
            speed = math.hypot(drone.vel_x, drone.vel_y)
            if speed < 2:
                drone.idle_frames += 1
            else:
                drone.idle_frames = 0

            if dist > TARGET_SIZE and drone.idle_frames > 60:
                ge[i].fitness -= 0.2  # Stała mała kara

            # --- FITNESS: Eksploracja (Zależna od trudności) ---
            if dist < drone.min_dist:
                # Im trudniejsza generacja, tym więcej wart każdy piksel zbliżenia
                improvement = drone.min_dist - dist
                if improvement > 0.5:
                    drone.frames_without_progress = 0  # Reset licznika
                    ge[i].fitness += (
                        improvement * 0.5 * difficulty_multiplier * time_decay
                    )
                drone.min_dist = dist
            else:
                drone.frames_without_progress += 1

            # --- FITNESS: Kolizje (Proporcjonalne) ---
            hit_wall, hit_obstacle = check_collisions(drone, obstacles)

            if hit_wall or hit_obstacle:
                # Zamiast odejmowania stałej, zabieramy % kapitału i mały bonus kary
                # Dron który nic nie zrobił, spadnie do 10-20 pkt.
                ge[i].fitness *= 0.5  # Tracisz połowę za zniszczenie sprzętu

                dist_ratio = max(0.1, min(dist / drone.initial_dist, 1.0))
                ge[i].fitness -= 20.0 * dist_ratio  # Dodatkowa kara zależna od dystansu

                drones_to_remove.append(i)

            else:
                # --- FITNESS: Cel (Hover) ---
                if dist < TARGET_SIZE:
                    drone.hover_frames += 1
                    # Nagroda za każdą klatkę w kółku, skalowana trudnością
                    hover_reward = (
                        3.0
                        * difficulty_multiplier
                        * time_decay
                        * (1 + drone.hover_frames * 0.1)
                    )
                    ge[i].fitness += hover_reward

                    if drone.hover_frames >= FPS * 1:
                        # Bonus za stabilność
                        ge[i].fitness += 2000 * difficulty_multiplier * time_decay
                        ge[i].fitness += (max_frames - frames) * 2
                        drones_to_remove.append(i)
                else:
                    if drone.frames_without_progress > stagnation_limit:
                        ge[i].fitness *= 0.9
                        drones_to_remove.append(i)
                        continue  # Przejdź do następnego drona
                    drone.hover_frames = 0

                if (
                    ge[i].fitness < 10.0
                ):  # Jeśli roztrwonił prawie cały startowy kapitał
                    drones_to_remove.append(i)

        for i in reversed(drones_to_remove):
            remove_drone(i, drones, nets, ge)

        if show_simulation:
            render_simulation(screen, drones, target_pos, obstacles)
            clock.tick(FPS)
        else:
            screen.fill((20, 25, 30))
            hidden_txt = STAT_FONT.render("SIMULATION HIDDEN", True, (255, 255, 255))
            text_rect = hidden_txt.get_rect(
                center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
            )
            screen.blit(hidden_txt, text_rect)

            pygame.display.flip()

    # --- OCENA KOŃCOWA ---
    for i, drone in enumerate(drones):
        # Drony, które przeżyły, ale nie dotarły
        dist = math.hypot(drone.x - target_pos[0], drone.y - target_pos[1])
        if dist > drone.initial_dist:
            ge[i].fitness *= 0.8  # Kara 20% za ucieczkę
        else:
            # Mały bonus za bycie bliżej niż na starcie
            ge[i].fitness *= 1.2


# =====================================================================
# NEAT I TRYB POKAZOWY
# =====================================================================


def run_neat(config_path: str, checkpoint: str | None = None) -> None:
    """Konfiguruje i uruchamia algorytm NEAT."""
    # Inicjujemy Pygame przed startem ewolucji
    pygame.init()
    pygame.font.init()
    pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("BIAI Drone Sim - AI Evolution")

    # Wczytanie konfiguracji z pliku
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    checkpoint_dir = Path("checkpoints")
    # Tworzenie populacji (np. 50 dronów)
    if checkpoint == "latest":
        if checkpoint_dir.exists():
            # Szukamy plików wewnątrz folderu 'checkpoints'
            checkpoints = [
                f
                for f in checkpoint_dir.iterdir()
                if f.is_file() and f.name.startswith("neat-checkpoint-")
            ]

            if checkpoints:
                # Sortujemy po numerze na końcu nazwy pliku
                latest_checkpoint_path = max(
                    checkpoints, key=lambda x: int(x.name.split("-")[-1])
                )
                # Musimy przekazać pełną ścieżkę jako string dla NEAT
                checkpoint = str(latest_checkpoint_path)
                print(f"Znaleziono najnowszy zapis: {checkpoint}")
            else:
                print("Folder 'checkpoints' jest pusty.")
                checkpoint = None
        else:
            print("Folder 'checkpoints' nie istnieje.")
            checkpoint = None

    # Tworzenie populacji na podstawie tego, co ustaliliśmy wyżej
    if checkpoint is not None:
        print(f"Wczytywanie stanu ewolucji z pliku: {checkpoint}")
        population = neat.Checkpointer.restore_checkpoint(checkpoint)
    else:
        print("Tworzenie nowej populacji od zera...")
        population = neat.Population(config)

    # Dodanie reporterów (wypisują statystyki do konsoli)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_prefix = str(checkpoint_dir / "neat-checkpoint-")
    population.add_reporter(neat.Checkpointer(20, filename_prefix=checkpoint_prefix))

    # START EWOLUCJI
    # Uruchamiamy na maksymalnie 100 generacji
    print("Starting neuroevolution...")
    winner = population.run(eval_genomes, EVOLUTION_CYCLES)

    # Po zakończeniu treningu
    print(f"\nBest genome found:\n{winner}")

    with open("best_drone.pkl", "wb") as f:
        pickle.dump(winner, f)
        print("Saved best genome to 'best_drone.pkl'")

    pygame.quit()


def test_best_drone(config_path: str, genome_path: str = "best_drone.pkl") -> None:
    """Wczytuje najlepszego drona z pliku i pozwala go przetestować."""
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    with open(genome_path, "rb") as f:
        winner_genome = pickle.load(f)

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("BIAI Drone Sim - CHAMPION")
    clock = pygame.time.Clock()

    net = neat.nn.FeedForwardNetwork.create(winner_genome, config)
    drone = Drone(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)

    # NOWOŚĆ: Generujemy startowe przeszkody, żeby dron miał co omijać!
    target_pos = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4)
    obstacles = generate_obstacles(target_pos, num_obstacles=8)

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            # Opcjonalnie: Prawy przycisk myszy odświeża układ przeszkód
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                obstacles = generate_obstacles(target_pos, num_obstacles=8)

        mouse_x, mouse_y = pygame.mouse.get_pos()
        target_pos = (mouse_x, mouse_y)

        # Używamy tej samej fizyki i renderowania co w treningu!
        inputs = drone.get_inputs(
            target_pos[0], target_pos[1], SCREEN_WIDTH, SCREEN_HEIGHT, obstacles
        )
        output = net.activate(inputs)
        drone.l_thrust = output[0]  # (output[0] + 1.0) / 2.0
        drone.r_thrust = output[1]  # (output[1] + 1.0) / 2.0
        drone.apply_thrust(drone.l_thrust, drone.r_thrust)
        drone.update()

        render_simulation(screen, [drone], target_pos, obstacles)
        clock.tick(FPS)

    pygame.quit()
