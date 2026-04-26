import neat
import pygame
import math
import sys
import random
import pickle
from dataclasses import dataclass

from pathlib import Path
from typing import cast, Any
from .drone import Drone
from .constants import *

pygame.font.init()
STAT_FONT = pygame.font.SysFont("arial", 50)

show_simulation = True


@dataclass
class EvolutionStats:
    """Przechowuje stan i postępy drona dla algorytmu NEAT."""

    initial_dist: float = 0.0
    min_dist: float = 0.0
    hover_frames: int = 0
    idle_frames: int = 0
    frames_without_progress: int = 0
    has_touched_target: bool = False
    accumulated_rotation: float = 0.0


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


def render_simulation(
    screen: pygame.Surface,
    drones: list[Drone],  # Lista obiektów klasy Drone (z fizyką SI)
    target_pos_px: tuple[int, int],
    obstacles: list[pygame.Rect],
    PPM: float,  # Nowy argument: Skala
) -> None:
    """Rysuje całą klatkę symulacji."""
    _ = screen.fill((20, 25, 30))

    # Rysowanie przeszkód
    for obs in obstacles:
        _ = pygame.draw.rect(screen, (150, 50, 50), obs)
        _ = pygame.draw.rect(screen, (255, 100, 100), obs, 2)

    # Rysowanie celu (Zakładamy, że TARGET_SIZE z constants.py jest w pikselach)
    _ = pygame.draw.circle(screen, (0, 255, 0), target_pos_px, TARGET_SIZE, 2)
    _ = pygame.draw.circle(screen, (0, 255, 0), target_pos_px, 3)

    # Przeliczamy pozycję celu na metry, by podać ją dronom
    target_pos_m = (target_pos_px[0] / PPM, target_pos_px[1] / PPM)

    # Rysowanie dronów
    for i, drone in enumerate(drones):
        # Flaga: Rysuj zaawansowane zmysły tylko dla pierwszego (lub jedynego) drona
        is_champion = i == 0

        drone.draw(
            screen=screen,
            target_pos_m=target_pos_m,
            PPM=PPM,
            show_radar=is_champion,  # Radar tylko dla lidera
            show_sensors=is_champion,  # Sensory tylko dla lidera
            show_thrust=True,  # Płomienie silników dla wszystkich!
            show_hitbox=False,
        )

    _ = pygame.display.flip()


def remove_drone(
    index: int,
    drones: list[Drone],
    stats: list[EvolutionStats],
    nets: list[neat.nn.FeedForwardNetwork],
    ge: list[neat.DefaultGenome],
) -> None:
    # remove from simulation
    _ = drones.pop(index)
    _ = stats.pop(index)
    _ = nets.pop(index)
    _ = ge.pop(index)


# =====================================================================
# GŁÓWNA LOGIKA EWOLUCJI
# =====================================================================
def is_target_visible(
    start_pos: tuple[int, int],
    target_pos: tuple[float, float],
    obstacles: list[pygame.Rect],
) -> bool:
    """Sprawdza, czy linia między startem a celem jest przecięta przez przeszkodę."""
    line = (start_pos, target_pos)
    for obs in obstacles:
        if obs.clipline(line):
            return False
    return True


def calculate_difficulty(
    start_pos_px: tuple[int, int],
    target_pos_px: tuple[float, float],
    obstacles: list[pygame.Rect],
) -> float:
    """Oblicza mnożnik trudności na podstawie widoczności i dystansu."""
    visible: bool = is_target_visible(start_pos_px, target_pos_px, obstacles)
    dist_px = math.hypot(
        target_pos_px[0] - start_pos_px[0], target_pos_px[1] - start_pos_px[1]
    )
    dist_factor = dist_px / (SCREEN_WIDTH / 2)

    return (1.5 if not visible else 1.0) * (1.0 + dist_factor)


def _update_fitness(
    drone: Drone,
    stats: EvolutionStats,
    genome: Any,
    target_pos_m: tuple[float, float],
    difficulty_multiplier: float,
    current_time_sec: float,
    time_decay: float,
) -> None:
    """Oblicza nagrody i kary dla pojedynczej klatki (Fizyka SI)."""

    t_peak = 2.0  # W której sekundzie nagroda jest największa (możesz przenieść do constants.py)

    # Zabezpieczenie na wypadek t_peak = 0
    if t_peak > 0:
        # Wzór matematyczny z wykresu powyżej
        time_multiplier = (current_time_sec / t_peak) * math.exp(
            1.0 - (current_time_sec / t_peak)
        )
    else:
        time_multiplier = 0.0

    survival_bonus = FIT_SURVIVAL_FRAME_REWARD * time_multiplier
    genome.fitness += survival_bonus

    # 1. Nagroda za płynność (Smoothness)
    jitter = abs(drone.l_command - drone.prev_l_command) + abs(
        drone.r_command - drone.prev_r_command
    )

    # smoothness = max(0.0, 1.0 - jitter)
    # genome.fitness += smoothness * FIT_SMOOTHNESS_MULT * difficulty_multiplier

    genome.fitness -= jitter * FIT_JITTER_PENALTY

    # 2. Dystans do celu (w metrach!)
    dist_m: float = math.hypot(drone._x - target_pos_m[0], drone._y - target_pos_m[1])

    # 3. Stabilność przy celu
    if dist_m < (TARGET_SIZE / PPM) * 2:
        stability = max(0.0, 1.0 - abs(drone._angular_vel / 5.0))
        genome.fitness += stability * FIT_STABILITY_MULT * difficulty_multiplier

    # 4. Discovery Bonus
    if not stats.has_touched_target and dist_m < (TARGET_SIZE / PPM):
        genome.fitness += FIT_DISCOVERY_BONUS * difficulty_multiplier * time_decay
        stats.has_touched_target = True

    # 5. Eksploracja i Stagnacja
    if dist_m < stats.min_dist:
        improvement = stats.min_dist - dist_m
        if improvement > 0.0025:  # ok 0.5px
            stats.frames_without_progress = 0
            genome.fitness += (
                improvement
                * PPM
                * FIT_EXPLORATION_MULT
                * difficulty_multiplier
                * time_decay
            )
        stats.min_dist = dist_m
    else:
        stats.frames_without_progress += 1

    # --- DETEKCJA "BĄCZKÓW" (SPIN DETECTION) ---
    # Pobieramy zmianę kąta w tej klatce (angular_vel jest w stopniach/klatkę)
    current_spin = drone._angular_vel

    # Sprawdzamy, czy dron zmienił kierunek obrotu (iloczyn ujemny oznacza różne znaki)
    if current_spin * stats.accumulated_rotation < 0:
        # Jeśli zaczął kręcić się w drugą stronę - resetujemy licznik "ciągłego spinu"
        stats.accumulated_rotation = 0.0

    # Dodajemy obecny obrót do puli
    stats.accumulated_rotation += current_spin

    # Jeśli przekroczył limit (np. 720 stopni w lewo lub w prawo)
    spin_threshold_deg = MAX_ALLOWED_SPINS * 360.0
    if abs(stats.accumulated_rotation) > spin_threshold_deg:
        # Ostra kara co klatkę, dopóki się kręci
        # To sprawi, że fitness drona błyskawicznie spadnie
        genome.fitness -= FIT_SPIN_PENALTY  # Zabieramy 5% punktów w każdej klatce spinu


def eval_genomes(
    genomes: list[tuple[int, neat.DefaultGenome]], config: neat.Config
) -> None:
    global show_simulation
    screen = pygame.display.get_surface()
    clock = pygame.time.Clock()

    for genome_id, genome in genomes:
        cast(
            Any, genome
        ).fitness = FIT_START_CAPITAL  # lub np. 0.0, jeśli użyjesz akumulacji

    # 2. Definiujemy nasze 3 rundy (Test Suite)
    scenarios: list[tuple[str, int]] = [
        ("Runda 1: Otwarte Niebo", 0),
        ("Runda 2: Standard", 5),
        ("Runda 3: Tor Przeszkód", 10),
    ]

    for round_name, num_obs in scenarios:
        saved_fitness = {genome_id: cast(Any, g).fitness for genome_id, g in genomes}
        nets: list[neat.nn.FeedForwardNetwork] = []
        ge: list[neat.DefaultGenome] = []
        drones: list[Drone] = []
        stats_list: list[EvolutionStats] = []
        # Setup środowiska
        target_px = (
            random.randint(100, SCREEN_WIDTH - 100),
            random.randint(100, SCREEN_HEIGHT - 100),
        )
        target_m: tuple[float, float] = (target_px[0] / PPM, target_px[1] / PPM)
        obstacles = generate_obstacles(target_px, num_obs)

        start_px = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        diff_mult = calculate_difficulty(start_px, target_px, obstacles)

        for _, genome in genomes:
            # 1. NEAT Setup
            genome_any = cast(Any, genome)
            genome_any.fitness = FIT_START_CAPITAL
            nets.append(neat.nn.FeedForwardNetwork.create(genome, config))

            # 2. Tworzenie fizycznego drona (w metrach)
            drone_x = start_px[0] / PPM
            drone_y = start_px[1] / PPM
            new_drone = Drone(drone_x, drone_y)

            # 3. Tworzenie statystyk ewolucyjnych dla tego konkretnego drona
            # Obliczamy dystans początkowy
            d_start = math.hypot(target_m[0] - drone_x, target_m[1] - drone_y)

            new_stats = EvolutionStats(initial_dist=d_start, min_dist=d_start)

            # 4. Dodawanie do list (kolejność musi być identyczna we wszystkich listach!)
            drones.append(new_drone)
            stats_list.append(new_stats)
            ge.append(genome)

        frames = 0
        max_frames = FPS * SIMULATION_TIME
        dt = 1.0 / FPS

        while frames < max_frames and drones:
            frames += 1
            current_time_sec = frames / FPS
            time_decay = 1.0 - (0.8 * (frames / max_frames))

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_h:
                    show_simulation = not show_simulation

            to_remove = []
            for i, drone in enumerate(drones):
                stats = stats_list[i]
                genome_any = cast(Any, ge[i])
                # 1. AI Decision
                inputs = drone.get_inputs(
                    target_m, SCREEN_WIDTH, SCREEN_HEIGHT, obstacles, PPM
                )
                output = nets[i].activate(inputs)

                # 2. Fizyka
                drone.set_engine_thrust(output[0], output[1])
                drone.update(dt)

                # 3. Fitness
                _ = _update_fitness(
                    drone,
                    stats,
                    ge[i],
                    target_m,
                    diff_mult,
                    current_time_sec,
                    time_decay,
                )

                dist_m: float = math.hypot(
                    drone._x - target_m[0], drone._y - target_m[1]
                )

                # 4. Kolizje i Stagnacja (Warunki usunięcia)
                if drone.check_collision(SCREEN_WIDTH, SCREEN_HEIGHT, obstacles, PPM):
                    genome_any.fitness *= FIT_CRASH_MULT
                    genome_any.fitness -= FIT_CRASH_BASE_PENALTY * max(
                        0.1, dist_m / stats.initial_dist
                    )
                    to_remove.append(i)
                    continue

                if dist_m < (TARGET_SIZE / PPM):
                    stats.hover_frames += 1
                    genome_any.fitness += (
                        FIT_HOVER_REWARD
                        * diff_mult
                        * time_decay
                        * (1 + stats.hover_frames * 0.1)
                    )
                    if stats.hover_frames >= FPS * HOVER_REQUIRED_SEC:
                        genome_any.fitness += (
                            FIT_HOVER_SUCCESS_BONUS * diff_mult * time_decay
                        )
                        genome_any.fitness += (max_frames - frames) * 2
                        to_remove.append(i)
                else:
                    stats.hover_frames = 0
                    if stats.frames_without_progress > FPS * STAGNATION_LIMIT_SEC:
                        genome_any.fitness *= FIT_STAGNATION_MULT
                        to_remove.append(i)

            # Usuwanie dronów
            for index in reversed(to_remove):
                remove_drone(index, drones, stats_list, nets, ge)

            # Render
            if show_simulation:
                render_simulation(screen, drones, target_px, obstacles, PPM)
                clock.tick(FPS)

        # Ocena końcowa dla ocalałych w rundzie
        for i, drone in enumerate(drones):
            stats = stats_list[i]
            dist_m = math.hypot(drone._x - target_m[0], drone._y - target_m[1])
            genome_any = cast(Any, ge[i])
            if dist_m > stats.initial_dist:
                genome_any.fitness *= FIT_ESCAPE_PENALTY_MULT
            else:
                genome_any.fitness *= FIT_SURVIVAL_BONUS_MULT

        # Koniec rundy! Dodajemy wynik z tej rundy do tego, co zapisaliśmy wcześniej
        for genome_id, genome in genomes:
            genome_any = cast(Any, genome)
            round_score = genome_any.fitness
            # Łączymy "bank" z poprzednich rund z tym, co ugrał w tej
            genome_any.fitness = saved_fitness[genome_id] + round_score

    # po wszystkich rundach całkowity fitness
    num_rounds = len(scenarios)
    for genome_id, genome in genomes:
        cast(Any, genome).fitness /= num_rounds


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

        mx, my = pygame.mouse.get_pos()
        target_px = (mx, my)
        target_m = (mx / PPM, my / PPM)  # Konwersja na metry dla AI

        # Używamy tej samej fizyki i renderowania co w treningu!
        inputs = drone.get_inputs(target_m, SCREEN_WIDTH, SCREEN_HEIGHT, obstacles, PPM)
        output = net.activate(inputs)
        drone.set_engine_thrust(output[0], output[1])
        drone.update()

        render_simulation(screen, [drone], target_px, obstacles, PPM)
        clock.tick(FPS)

    pygame.quit()
