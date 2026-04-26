import neat
from neat.nn import FeedForwardNetwork
from numpy import append, true_divide
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
from .pathfinding import get_expert_path

pygame.font.init()
STAT_FONT = pygame.font.SysFont("arial", 50)

show_simulation = True
generation_count = 0


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


def _update_and_eval_drone(
    current_frame: int,
    dt: float,
    drone: Drone,
    target_m: tuple[float, float],
    stats: EvolutionStats,
    genome: neat.DefaultGenome,
    net: FeedForwardNetwork,
    expert: HardcodedBrain,
    help_weight: float,
    obstacles: list[pygame.Rect],
    difficulty_multiplier: float,
) -> bool:
    current_time = current_frame * dt
    to_remove = False

    genome_any = cast(Any, genome)

    # get inpputs from drone sensors and internal states
    state_inputs = drone.get_inputs(
        target_pos_m=target_m,
        screen_width_px=SCREEN_WIDTH,
        screen_height_px=SCREEN_HEIGHT,
        obstacles=obstacles,
        PPM=PPM,
    )

    net_action = net.activate(state_inputs)
    # expert_action = expert.activate(drone, target_m)

    # 2. PORÓWNANIE Z EKSPERTEM (Imitation Learning)
    # if help_weight > 0.0:
    #    # Obliczamy różnicę między tym co zrobiła sieć, a co zrobiłby ekspert (np. błąd średniokwadratowy)
    #    action_diff = sum((n - e) ** 2 for n, e in zip(net_action, expert_action))
    #    # Nakładamy karę za odchylenie, skalowaną przez help_weight i dt
    #    genome_any.fitness -= action_diff * FIT_EXPERT_PENALTY_MULT * help_weight * dt

    # 3. RUCH DRONA (Bardzo ważne, bez tego dron stoi!)
    drone.set_engine_thrust(net_action[0], net_action[1])
    drone.update(dt)

    dist_m: float = math.hypot(drone._x - target_m[0], drone._y - target_m[1])

    # escape early check
    if dist_m > stats.initial_dist_m + 2.0:
        # genome_any.fitness *= 1 - FIT_ESCAPE_PENALTY_PERC
        return True

    # exploration bonus
    if dist_m < stats.min_dist_m:
        improvement = stats.min_dist_m - dist_m
        # around 200m from target multiplier starts raising noticeably
        dist_multiplier = time_multiplier = 1.0 + (400.0 / (200 + dist_m))
        if improvement > FIT_STAGNATION_DISTANCE_LIMIT_M:
            stats.time_without_progress = 0
            genome_any.fitness += improvement * FIT_EXPLORATION_MULT * dist_multiplier
        stats.min_dist_m = dist_m
    else:
        stats.time_without_progress += dt

    # check collision
    if drone.check_collision(SCREEN_WIDTH, SCREEN_HEIGHT, obstacles, PPM):
        dist_coeff = max(0.5, min(1.0, dist_m / stats.initial_dist_m))
        crash_speed = math.hypot(drone._vel_x, drone._vel_y)
        kamikaze_mult = max(1.0, crash_speed / SAFE_CRASH_SPEED_M_S)
        # calculate penalty based on speed of collision
        actual_penalty_perc = min(
            0.95, FIT_CRASH_PENALTY_PERC * dist_coeff * kamikaze_mult
        )
        # if closer to target (dist_coeff lower) apply less penalty min, half (for now)
        genome_any.fitness *= 1 - actual_penalty_perc
        # less penalty if it crashed closer to the point
        genome_any.fitness -= FIT_CRASH_BASE_PENALTY * dist_coeff
        # apply penalty if above safe speed
        if crash_speed > SAFE_CRASH_SPEED_M_S:
            genome_any.fitness -= FIT_KAMIKAZE_PENALTY * (
                crash_speed - SAFE_CRASH_SPEED_M_S
            )
        to_remove = True

    if dist_m < (TARGET_SIZE_PX / PPM):
        stats.hover_time += dt
        genome_any.fitness += (
            dt * FIT_HOVER_REWARD * (1 + stats.hover_time * 0.1) * difficulty_multiplier
        )
        time_multiplier = 1.0 + (2.0 / (1.0 + current_time))
        if stats.hover_time >= HOVER_REQUIRED_SEC:
            # Obliczamy nieliniowy mnożnik czasu (K=2.0)
            genome_any.fitness += (
                FIT_HOVER_SUCCESS_BONUS * time_multiplier * difficulty_multiplier
            )
            to_remove = True
        if not stats.has_touched_target:
            stats.has_touched_target = True
            genome_any.fitness += FIT_DISCOVERY_BONUS * time_multiplier
    else:
        stats.hover_time = 0
        if stats.time_without_progress > STAGNATION_LIMIT_SEC:
            genome_any.fitness *= 1 - FIT_STAGNATION_PENALTY_PERC
            to_remove = True

    return to_remove


def eval_genomes(
    genomes: list[tuple[int, neat.DefaultGenome]], config: neat.Config
) -> None:
    global show_simulation
    global generation_count
    screen = pygame.display.get_surface()
    clock = pygame.time.Clock()

    max_help_gens = 50
    help_weight = max(0, 1.0 - (generation_count / max_help_gens))

    for genome_id, genome in genomes:
        cast(
            Any, genome
        ).fitness = FIT_START_CAPITAL  # lub np. 0.0, jeśli użyjesz akumulacji

    # 2. Definiujemy nasze 3 rundy (Test Suite)
    scenarios: list[tuple[str, int]] = [
        ("Runda 1: Otwarte Niebo", 0),
        ("Runda 2: Standard", 2),
        ("Runda 3: Tor Przeszkód", 4),
    ]

    expert = HardcodedBrain()

    for round_name, num_obs in scenarios:
        saved_fitness = {genome_id: cast(Any, g).fitness for genome_id, g in genomes}
        nets: list[neat.nn.FeedForwardNetwork] = []
        ge: list[neat.DefaultGenome] = []
        drones: list[Drone] = []
        stats_list: list[EvolutionStats] = []

        # 'expert' drone that already knows how to fly
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

        current_frame = 0
        max_frames = FPS * SIMULATION_TIME
        dt = 1.0 / FPS

        # finding difficulty multiplier
        expert_path: list[tuple[int, int]] = []
        valid_map_found = False
        max_retries = 50  # Zabezpieczenie przed nieskończoną pętlą
        attempts = 0

        while not valid_map_found and attempts < max_retries:
            attempts += 1

            # 1. Losujemy pozycje
            start_px, target_px = generate_start_and_target(
                SCREEN_WIDTH, SCREEN_HEIGHT, MAP_MARGIN_PX, MIN_SPAWN_DISTANCE_PX
            )
            # 2. Losujemy przeszkody
            obstacles = generate_obstacles(start_px, target_px, num_obs)

            # 3. Pytamy eksperta, czy da się to w ogóle przelecieć
            expert_path = get_expert_path(
                start_px, target_px, obstacles, drone_radius_px=15
            )

            if expert_path:
                valid_map_found = True

        if not valid_map_found:
            print(
                f"Ostrzeżenie: Nie udało się wygenerować mapy po {max_retries} próbach! Lecimy na pustej mapie."
            )
            obstacles = []  # Awaryjne wyczyszczenie mapy
            expert_path = [start_px, target_px]  # Ścieżka prosta

        # Odległość w linii prostej
        straight_dist = math.hypot(
            target_px[0] - start_px[0], target_px[1] - start_px[1]
        )

        # Obliczamy faktyczną długość wyznaczonej ścieżki
        path_length = 0
        current_pt = start_px
        for pt in expert_path:
            path_length += math.hypot(pt[0] - current_pt[0], pt[1] - current_pt[1])
            current_pt = pt

        # Wyliczamy mnożnik trudności (Tortuosity)
        # Jeśli ścieżka omija przeszkody, będzie dłuższa niż prosta, np. stosunek 1.5
        diff_mult = max(1.0, path_length / straight_dist)

        # loop over drones each frame
        while current_frame < max_frames and drones:
            current_frame += 1
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_h:
                    show_simulation = not show_simulation

            to_remove = []
            for i, drone in enumerate(drones):
                # Evaluate single drone, returns true if should be removed - either reached end or crashed etc.
                should_remove = _update_and_eval_drone(
                    current_frame=current_frame,
                    dt=dt,
                    drone=drone,
                    target_m=target_m,
                    stats=stats_list[i],
                    genome=ge[i],
                    net=nets[i],
                    expert=expert,
                    help_weight=help_weight,
                    obstacles=obstacles,
                    difficulty_multiplier=diff_mult,
                )
                if should_remove:
                    to_remove.append(i)

            for index in reversed(to_remove):
                # todo naliczyć premię za szybszy czas - mnożnik?
                remove_drone(index, drones, stats_list, nets, ge)

            # Render
            if show_simulation:
                render_simulation(screen, drones, target_px, obstacles, PPM)
                clock.tick(FPS)

        # Koniec rundy! Dodajemy wynik z tej rundy do tego, co zapisaliśmy wcześniej
        # todo - ewentualnie naliczyć premie za trudność - mnożnik na podstawie eksperta albo inny
        for genome_id, genome in genomes:
            genome_any = cast(Any, genome)
            round_score = genome_any.fitness
            # Łączymy "bank" z poprzednich rund z tym, co ugrał w tej
            genome_any.fitness = saved_fitness[genome_id] + round_score

    # po wszystkich rundach całkowity fitness
    num_rounds = len(scenarios)
    for genome_id, genome in genomes:
        cast(Any, genome).fitness /= num_rounds
    generation_count += 1


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


def reset_test_drone(target_m: tuple[float, float]):
    """Pomocnicza funkcja do tworzenia świeżych obiektów po resecie."""
    new_drone = Drone((SCREEN_WIDTH // 2) / PPM, (SCREEN_HEIGHT // 2) / PPM)
    d_start = math.hypot(target_m[0] - new_drone._x, target_m[1] - new_drone._y)

    # Używamy nazw atrybutów zdefiniowanych w EvolutionStats
    new_stats = EvolutionStats(initial_dist_m=d_start, min_dist_m=d_start)

    class DummyGenome:
        fitness = FIT_START_CAPITAL  #

    return new_drone, new_stats, DummyGenome()


def test_baseline() -> None:
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("BIAI Drone Sim - HARDCODED BASELINE")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 24)

    brain = HardcodedBrain()

    # Cel początkowy
    target_px = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4)
    target_m = (target_px[0] / PPM, target_px[1] / PPM)
    obstacles = []

    drone, stats, dummy_genome = reset_test_drone(target_m)

    frames = 0
    max_frames = FPS * SIMULATION_TIME  #
    run = True

    while run:
        dt = 1.0 / FPS
        frames += 1
        current_time_sec = frames / FPS
        # time_decay analogiczny do tego w eval_genomes
        time_decay = 1.0 - (0.8 * (frames / max_frames)) if max_frames > 0 else 1.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                drone_pos_px = (int(drone._x * PPM), int(drone._y * PPM))
                obstacles = generate_obstacles(drone_pos_px, target_px, num_obstacles=5)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                target_px = pygame.mouse.get_pos()
                target_m = (target_px[0] / PPM, target_px[1] / PPM)
                # Restartujemy statystyki odległości dla nowego celu
                d_curr = math.hypot(target_m[0] - drone._x, target_m[1] - drone._y)
                stats.initial_dist_m = d_curr
                stats.min_dist_m = d_curr
        # 1. Sensory i Myślenie
        _ = drone.get_sensor_data(SCREEN_WIDTH, SCREEN_HEIGHT, obstacles, PPM)  #
        output = brain.activate(drone, target_m)
        drone.set_engine_thrust(output[0], output[1])

        # 2. Fizyka
        drone.update(dt)  #
        dist_m = math.hypot(drone._x - target_m[0], drone._y - target_m[1])

        # 3. Aktualizacja Fitness (zgodnie z sygnaturą w evolution.py)
        # diff_mult ustawiamy na 1.0 dla prostoty testu baseline
        # _update_fitness(
        #    drone,
        #    stats,
        #    dummy_genome,
        #    target_m,
        #    dist_m=dist_m,
        #    difficulty_multiplier=1.0,
        #    current_time_sec=current_time_sec,
        #    dt=dt,
        #    time_decay=time_decay,
        # )

        # 4. Warunki końca / resetu
        is_crashed = drone.check_collision(SCREEN_WIDTH, SCREEN_HEIGHT, obstacles, PPM)

        # Sukces (Hovering) - używamy klatek, bo tak liczy EvolutionStats
        is_success = False
        if dist_m < (TARGET_SIZE_PX / PPM):
            stats.hover_time += dt
            if stats.hover_time >= HOVER_REQUIRED_SEC:  #
                dummy_genome.fitness += FIT_HOVER_SUCCESS_BONUS  #
                is_success = True
        else:
            stats.hover_time = 0

        # Stagnacja
        is_stagnated = stats.time_without_progress > STAGNATION_LIMIT_SEC

        if is_crashed or is_success or is_stagnated or frames >= max_frames:
            status = (
                "KOLIZJA" if is_crashed else "SUKCES" if is_success else "TIMEOUT/STAG"
            )
            print(f"--- KONIEC PRÓBY: {status} ---")
            print(
                f"Final Fitness: {dummy_genome.fitness:.1f} | Czas: {current_time_sec:.1f}s"
            )

            # Reset symulacji
            drone, stats, dummy_genome = reset_test_drone(target_m)
            frames = 0
            continue

        # --- RYSOWANIE ---
        render_simulation(screen, [drone], target_px, obstacles, PPM)  #

        # UI Overlay
        txt_fit = font.render(
            f"Fitness: {dummy_genome.fitness:.1f}", True, (255, 255, 0)
        )
        txt_dist = font.render(f"Dystans: {dist_m:.2f} m", True, (0, 255, 255))
        txt_time = font.render(f"Czas: {current_time_sec:.1f} s", True, (255, 255, 255))
        screen.blit(txt_fit, (10, 10))
        screen.blit(txt_dist, (10, 40))
        screen.blit(txt_time, (10, 70))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
