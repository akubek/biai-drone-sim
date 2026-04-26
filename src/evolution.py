import neat
import pygame
import math
import sys
import random
import pickle
from dataclasses import dataclass

from pathlib import Path
from typing import cast, Any

from pygame.math import clamp
from .drone import Drone
from .constants import *
from .hardcoded_brain import HardcodedBrain

pygame.font.init()
STAT_FONT = pygame.font.SysFont("arial", 50)

show_simulation = True


@dataclass
class EvolutionStats:
    """Przechowuje stan i postępy drona dla algorytmu NEAT."""

    initial_dist_m: float = 0.0
    min_dist_m: float = 0.0
    hover_time: float = 0.0
    idle_time: float = 0.0
    time_without_progress: float = 0.0
    has_touched_target: bool = False
    accumulated_rotation: float = 0.0


# =====================================================================
# METODY POMOCNICZE (ŚRODOWISKO I GRAFIKA)
# =====================================================================


# DODAJEMY start_pos jako argument!
def generate_obstacles(
    start_pos: tuple[int, int], target_pos: tuple[int, int], num_obstacles: int = 5
) -> list[pygame.Rect]:
    """Generuje losowe przeszkody, upewniając się, że nie blokują startu ani celu."""
    obstacles = []

    # POPRAWKA: Używamy przekazanego start_pos, a nie SCREEN_WIDTH // 2
    start_rect = pygame.Rect(start_pos[0] - 50, start_pos[1] - 50, 100, 100)
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


def generate_start_and_target(
    width: int, height: int, margin: int, min_dist: float
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Generuje losowy punkt startowy i docelowy gwarantując,
    że są od siebie oddalone o co najmniej min_dist.
    """
    # 1. Losujemy start gdziekolwiek (z zachowaniem marginesu od ściany)
    start_x = random.randint(margin, width - margin)
    start_y = random.randint(margin, height - margin)
    start_pos = (start_x, start_y)

    # 2. Losujemy cel dopóki nie będzie wystarczająco daleko
    while True:
        target_x = random.randint(margin, width - margin)
        target_y = random.randint(margin, height - margin)
        target_pos = (target_x, target_y)

        dist = math.hypot(target_x - start_x, target_y - start_y)

        if dist >= min_dist:
            return start_pos, target_pos


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
    _ = pygame.draw.circle(screen, (0, 255, 0), target_pos_px, TARGET_SIZE_PX, 2)
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

    return (
        (1.5 if not visible else 1.0) * (1.0 + dist_factor) * (1 + len(obstacles) / 16)
    )


def _update_fitness(
    drone: Drone,
    stats: EvolutionStats,
    genome: Any,
    target_m: tuple[float, float],
    dist_m: float,
    difficulty_multiplier: float,
    current_time_sec: float,
    dt: float,
    time_decay: float,
) -> None:
    """Oblicza nagrody i kary dla pojedynczej klatki (Fizyka SI)."""

    # Zabezpieczenie na wypadek t_peak = 0
    if SURVIVAL_TIME_PEAK > 0:
        # Wzór matematyczny z wykresu powyżej
        time_multiplier = (current_time_sec / SURVIVAL_TIME_PEAK) * math.exp(
            1.0 - (current_time_sec / SURVIVAL_TIME_PEAK)
        )
    else:
        time_multiplier = 0.0

    if current_time_sec < SURVIVAL_TIME_PEAK + 1.0:
        survival_bonus = FIT_SURVIVAL_FRAME_REWARD * time_multiplier
        genome.fitness += survival_bonus * dt

    # 1. Nagroda za płynność (Smoothness)
    jitter = abs(drone.l_command - drone.prev_l_command) + abs(
        drone.r_command - drone.prev_r_command
    )

    # smoothness = max(0.0, 1.0 - jitter)
    # genome.fitness += smoothness * FIT_SMOOTHNESS_MULT * difficulty_multiplier

    genome.fitness -= jitter * FIT_JITTER_PENALTY

    # 3. Stabilność przy celu
    if dist_m < (TARGET_SIZE_PX / PPM) * 2:
        stability = max(0.0, 1.0 - abs(drone._angular_vel / 5.0))
        genome.fitness += dt * stability * FIT_STABILITY_MULT * difficulty_multiplier

    # 4. Discovery Bonus
    if not stats.has_touched_target and dist_m < (TARGET_SIZE_PX / PPM):
        genome.fitness += FIT_DISCOVERY_BONUS * difficulty_multiplier * time_decay
        stats.has_touched_target = True

    # 5. Eksploracja i Stagnacja
    if dist_m < stats.min_dist_m:
        improvement = stats.min_dist_m - dist_m
        if improvement > FIT_STAGNATION_DISTANCE_LIMIT_M:
            stats.time_without_progress = 0
            genome.fitness += (
                improvement * FIT_EXPLORATION_MULT * difficulty_multiplier * time_decay
            )
        stats.min_dist_m = dist_m
    else:
        stats.time_without_progress += dt

    speed = math.hypot(drone._vel_x, drone._vel_y)

    # Próg lenistwa: np. 0.1 metra (zależy od Twoich jednostek prędkości,
    # jeśli _vel_x to m/s, to 0.1 m/s to bardzo powolny dryf)
    if speed < IDLE_MIN_SPEED:
        stats.idle_time += dt
    else:
        stats.idle_time = 0

    # Jeśli dron "wisi" bez sensu dłużej niż sekundę i NIE jest w celu:
    if stats.idle_time > IDLE_LIMIT_SEC and dist_m > (TARGET_SIZE_PX / PPM):
        # FIT_IDLE_PENALTY to np. 0.2 w constants.py
        genome.fitness -= dt * FIT_IDLE_PENALTY / difficulty_multiplier

    # --- NAGRODA ZA KIERUNKOWY WEKTOR PRĘDKOŚCI (VELOCITY ALIGNMENT) ---
    dx = target_m[0] - drone._x
    dy = target_m[1] - drone._y

    if dist_m > 0:
        # 1. Znormalizowany wektor kierunku do celu (wskazuje idealną drogę)
        dir_x = dx / dist_m
        dir_y = dy / dist_m

        # 2. Prędkość drona w metrach na sekundę
        vel_x_ms = (drone._vel_x * FPS) / PPM
        vel_y_ms = (drone._vel_y * FPS) / PPM

        # 3. Iloczyn skalarny: rzutowanie prędkości na idealny kierunek
        # Zwraca wartość w m/s. Jeśli leci prosto w cel -> max, bokiem -> 0, w tył -> ujemne
        velocity_towards_target = (vel_x_ms * dir_x) + (vel_y_ms * dir_y)

        if velocity_towards_target > 0:
            # Nagroda rośnie im szybciej i prościej leci
            genome.fitness += (
                velocity_towards_target
                * dt
                * FIT_DIR_VELOCITY_REWARD
                * difficulty_multiplier
            )
        else:
            # Kara za bezsensowny dryf w złą stronę (np. spadanie w dół gdy cel jest u góry)
            # velocity_towards_target jest tu ujemne, więc DODAJEMY je do fitnessu (zmniejszając go)
            genome.fitness += velocity_towards_target * dt * FIT_WRONG_DIR_PENALTY

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
        ("Runda 2: Standard", 2),
        ("Runda 3: Tor Przeszkód", 5),
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
        start_px, target_px = generate_start_and_target(
            SCREEN_WIDTH, SCREEN_HEIGHT, MAP_MARGIN_PX, MIN_SPAWN_DISTANCE_PX
        )
        target_m: tuple[float, float] = (target_px[0] / PPM, target_px[1] / PPM)
        obstacles = generate_obstacles(start_px, target_px, num_obs)
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
            # Obliczamy dystans początkowy w metrach
            d_start = math.hypot(target_m[0] - drone_x, target_m[1] - drone_y)

            new_stats = EvolutionStats(initial_dist_m=d_start, min_dist_m=d_start)

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

                dist_m: float = math.hypot(
                    drone._x - target_m[0], drone._y - target_m[1]
                )

                # zauwazamy ze dron od razu leci w zlym kierunku
                if dist_m > stats.initial_dist_m + 2.0:
                    genome_any.fitness *= 1 - FIT_ESCAPE_PENALTY_PERC
                    to_remove.append(i)
                    continue

                # 3. Fitness
                _ = _update_fitness(
                    drone,
                    stats,
                    ge[i],
                    target_m,
                    dist_m,
                    diff_mult,
                    current_time_sec,
                    dt,
                    time_decay,
                )

                # 4. Kolizje i Stagnacja (Warunki usunięcia)
                if drone.check_collision(SCREEN_WIDTH, SCREEN_HEIGHT, obstacles, PPM):
                    dist_coeff = max(0.5, min(1.0, dist_m / stats.initial_dist_m))

                    crash_speed = math.hypot(drone._vel_x, drone._vel_y)

                    kamikaze_mult = max(1.0, crash_speed / SAFE_CRASH_SPEED_M_S)

                    actual_penalty_perc = min(
                        0.95, FIT_CRASH_PENALTY_PERC * dist_coeff * kamikaze_mult
                    )
                    # if closer to target (dist_coeff lower) apply less penalty min, half (for now)
                    genome_any.fitness *= 1 - actual_penalty_perc
                    # less penalty if it crashed closer to the point
                    genome_any.fitness -= FIT_CRASH_BASE_PENALTY * dist_coeff

                    if crash_speed > SAFE_CRASH_SPEED_M_S:
                        genome_any.fitness -= FIT_KAMIKAZE_PENALTY * (
                            crash_speed - SAFE_CRASH_SPEED_M_S
                        )

                    to_remove.append(i)
                    continue

                if dist_m < (TARGET_SIZE_PX / PPM):
                    stats.hover_time += dt
                    genome_any.fitness += (
                        dt
                        * FIT_HOVER_REWARD
                        * diff_mult
                        * time_decay
                        * (1 + stats.hover_time * 0.1)
                    )
                    if stats.hover_time >= HOVER_REQUIRED_SEC:
                        genome_any.fitness += (
                            FIT_HOVER_SUCCESS_BONUS * diff_mult * time_decay
                        )
                        genome_any.fitness += (max_frames - frames) * 2
                        to_remove.append(i)
                else:
                    stats.hover_time = 0
                    if stats.time_without_progress > STAGNATION_LIMIT_SEC:
                        genome_any.fitness *= 1 - FIT_STAGNATION_PENALTY_PERC
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
            if dist_m > stats.initial_dist_m:
                genome_any.fitness *= 1 - FIT_ESCAPE_PENALTY_PERC
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
    drone = Drone((SCREEN_WIDTH // 2) / PPM, (SCREEN_HEIGHT // 2) / PPM)

    # NOWOŚĆ: Generujemy startowe przeszkody, żeby dron miał co omijać!
    target_pos = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4)
    drone_pos_px: tuple[int, int] = cast(
        tuple[int, int], (drone._x * PPM, drone._y * PPM)
    )
    obstacles = generate_obstacles(drone_pos_px, target_pos, num_obstacles=2)

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            # Opcjonalnie: Prawy przycisk myszy odświeża układ przeszkód
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                drone_pos_px: tuple[int, int] = cast(
                    tuple[int, int], (drone._x * PPM, drone._y * PPM)
                )
                obstacles = generate_obstacles(
                    drone_pos_px, target_pos, num_obstacles=2
                )

        mx, my = pygame.mouse.get_pos()
        target_px = (mx, my)
        target_m = (mx / PPM, my / PPM)  # Konwersja na metry dla AI

        # Używamy tej samej fizyki i renderowania co w treningu!
        inputs = drone.get_inputs(target_m, SCREEN_WIDTH, SCREEN_HEIGHT, obstacles, PPM)
        output = net.activate(inputs)
        drone.set_engine_thrust(output[0], output[1])
        drone.update(1.0 / FPS)

        render_simulation(screen, [drone], target_px, obstacles, PPM)
        clock.tick(FPS)

    pygame.quit()


def test_baseline() -> None:
    """Odpala drona sterowanego ręcznym algorytmem, by zbadać działanie fitnessu."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("BIAI Drone Sim - HARDCODED BASELINE")
    clock = pygame.time.Clock()

    # Inicjalizacja mózgu i drona
    brain = HardcodedBrain()
    drone = Drone((SCREEN_WIDTH // 2) / PPM, (SCREEN_HEIGHT // 2) / PPM)

    target_pos = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4)
    obstacles = []  # Pusty układ na start

    # Startowe metryki do fitnessu
    d_start = math.hypot(target_pos[0] / PPM - drone._x, target_pos[1] / PPM - drone._y)
    stats = EvolutionStats(initial_dist_m=d_start, min_dist_m=d_start)

    # Atrapa genomu (klasa, do której będziemy zapisywać zmienną .fitness)
    class DummyGenome:
        fitness = FIT_START_CAPITAL

    dummy_genome = DummyGenome()

    frames = 0
    run = True

    while run:
        dt = 1.0 / FPS
        frames += 1
        current_time_sec = frames / FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            # Prawy przycisk myszy losuje przeszkody, żebyś mógł sprawdzić jak w nie uderza
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                drone_pos_px: tuple[int, int] = cast(
                    tuple[int, int], (drone._x * PPM, drone._y * PPM)
                )
                obstacles = generate_obstacles(
                    drone_pos_px, target_pos, num_obstacles=4
                )

        # Cel podąża za kursorem
        mx, my = pygame.mouse.get_pos()
        target_px = (mx, my)
        target_m = (mx / PPM, my / PPM)

        # --- AI MYŚLI ---
        output = brain.activate(drone, target_m)

        # Fizyka
        drone.set_engine_thrust(output[0], output[1])
        drone.update(dt)

        # Obliczamy fitness co klatkę!
        dist_m = math.hypot(drone._x - target_m[0], drone._y - target_m[1])
        _update_fitness(
            drone=drone,
            stats=stats,
            genome=dummy_genome,
            dist_m=dist_m,
            target_m=target_m,
            difficulty_multiplier=1.0,
            current_time_sec=current_time_sec,
            dt=dt,
            time_decay=1.0,
        )

        # Jeśli dron się rozbił, resetujemy go (żebyś nie musiał restartować programu)
        if drone.check_collision(SCREEN_WIDTH, SCREEN_HEIGHT, obstacles, PPM):
            print(
                f"BUM! Końcowy fitness przed zniszczeniem: {dummy_genome.fitness:.1f}"
            )
            drone = Drone((SCREEN_WIDTH // 2) / PPM, (SCREEN_HEIGHT // 2) / PPM)
            dummy_genome.fitness = FIT_START_CAPITAL
            stats.min_dist_m = math.hypot(
                target_m[0] - drone._x, target_m[1] - drone._y
            )
            frames = 0  # Reset czasu

        render_simulation(screen, [drone], target_px, obstacles, PPM)

        # Wypisywanie wyniku co 10 sekund (co 60 klatek)
        if (frames / 10) % FPS == 0:
            print(
                f"T: {current_time_sec:.1f}s | Fitness: {dummy_genome.fitness:.1f} | Dystans: {dist_m:.2f}m"
            )

        clock.tick(FPS)

    pygame.quit()
