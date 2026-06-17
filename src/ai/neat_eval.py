import neat
from neat.nn import FeedForwardNetwork
from numpy import append, true_divide
import pygame
import math
import sys
import random
import pickle
import multiprocessing
from dataclasses import dataclass

from pathlib import Path
from typing import cast, Any

from pygame.math import clamp

from src.core.flight_controller import FlightController
from src.core.drone import Drone
from src.ai.expert import HardcodedBrain
from src.pathfinding import get_expert_path
from src.core.stats import EvolutionStats
from src.core.environment import generate_obstacles, generate_start_and_target
from src.utils.renderer import render_simulation

from src.config.config import *
from src.config.evolution import *
from src.config.physics import *
from src.config.rewards import *

pygame.font.init()
STAT_FONT = pygame.font.SysFont("arial", 50)

show_simulation = True
generation_count = 0

# ZMIENNE DO MIĘKKIEGO PRZEŁĄCZANIA (w trybie wizualnym)
render_graphics = True
target_fps = FPS
uncapped = False

USE_FLIGHT_CONTROLLER = False
global_flight_controller = FlightController()

# =====================================================================
# METODY POMOCNICZE (ŚRODOWISKO I EWALUACJA)
# =====================================================================
def _setup_population(config_path: str, checkpoint: str | None) -> tuple[neat.Population, neat.Config]:
    """Wspólna funkcja wczytująca konfigurację, checkpointy i reporterów."""
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path
    )

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # 1. Logika szukania najnowszego checkpointu ("latest")
    if checkpoint == "latest":
        checkpoints = [
            f for f in checkpoint_dir.iterdir()
            if f.is_file() and f.name.startswith("neat-checkpoint-")
        ]
        if checkpoints:
            latest_checkpoint_path = max(checkpoints, key=lambda x: int(x.name.split("-")[-1]))
            checkpoint = str(latest_checkpoint_path)
            print(f"Znaleziono najnowszy zapis: {checkpoint}")
        else:
            print("Folder 'checkpoints' jest pusty. Zaczynamy od zera.")
            checkpoint = None

    # 2. Tworzenie populacji
    if checkpoint is not None:
        print(f"Wczytywanie stanu ewolucji z pliku: {checkpoint}")
        population = neat.Checkpointer.restore_checkpoint(checkpoint)
    else:
        print("Tworzenie nowej populacji od zera...")
        population = neat.Population(config)

    # 3. Reporterzy (Wypisywanie w konsoli i zapisywanie plików)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    checkpoint_prefix = str(checkpoint_dir / "neat-checkpoint-")
    population.add_reporter(neat.Checkpointer(20, filename_prefix=checkpoint_prefix))

    return population, config

def _prepare_drone_and_stats(
    genome: neat.DefaultGenome, 
    config: neat.Config, 
    start_px: tuple[int, int], 
    target_px: tuple[int, int], 
    PPM: float
) -> tuple[neat.nn.FeedForwardNetwork, Drone, EvolutionStats]:
    """Tworzy sieć, fizycznego drona i inicjalizuje statystyki z limitami."""
    
    # 1. Sieć NEAT
    genome_any = cast(Any, genome)
    genome_any.fitness = FIT_START_CAPITAL
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # 2. Fizyczny Dron
    drone_x, drone_y = start_px[0] / PPM, start_px[1] / PPM
    drone = Drone(drone_x, drone_y)

    # 3. Matematyka dystansów i tolerancji ucieczki
    target_m = (target_px[0] / PPM, target_px[1] / PPM)
    d_start = math.hypot(target_m[0] - drone_x, target_m[1] - drone_y)
    
    # Przekątna świata i dozwolony margines (np. dystans + 30% przekątnej mapy)
    world_diagonal_m = math.hypot(SCREEN_WIDTH / PPM, SCREEN_HEIGHT / PPM)
    allowed_escape_dist = d_start + (world_diagonal_m * 0.3)

    # 4. Statystyki
    stats = EvolutionStats(
        initial_dist_m=d_start, 
        min_dist_m=d_start,
        last_stagnation_dist_m=d_start,
        max_hover_time_achieved=0.0,
        max_allowed_escape_dist_m=allowed_escape_dist
    )

    return net, drone, stats

def _remove_drone(
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


def apply_fitness_rules(
        drone: Drone, 
        stats: EvolutionStats, 
        genome: Any, 
        target_m: tuple[float, float], 
        dt: float, 
        obstacles: list, 
        difficulty_multiplier: float = 1.0,
        SCREEN_WIDTH: int = SCREEN_WIDTH,
        SCREEN_HEIGHT: int = SCREEN_HEIGHT,
        PPM: float = PPM

        ) -> tuple[bool, bool]:
    """Nalicza punkty i zwraca czy dron osiagnał sukces, czy się rozbił/utknał"""
    to_remove = False
    success = False
    dist_m = math.hypot(drone._x - target_m[0], drone._y - target_m[1])
    genome_any = cast(Any, genome)

    dist_m: float = math.hypot(drone._x - target_m[0], drone._y - target_m[1])

    # escape early check
    if dist_m > stats.max_allowed_escape_dist_m:
        return success, True

    # siponout check
    if abs(drone._angular_vel) > MAX_SAFE_ANGULAR_VEL:
        stats.spinout_time += dt
        if stats.spinout_time > MAX_ALLOWED_SPINOUT_TIME:
            return success, True
    else:
        stats.spinout_time = 0

    # exploration bonus
    if dist_m < stats.min_dist_m:
        improvement = stats.min_dist_m - dist_m
        stats.min_dist_m = dist_m
        # around 1m from target multiplier starts raising noticeably

        # the closer to the target the more points for progress
        dist_multiplier = 1.0 + (2.0 / (1.0 + dist_m))
        genome_any.fitness += improvement * FIT_EXPLORATION_MULT * dist_multiplier
    
    #stagnation check
    if (stats.last_stagnation_dist_m - dist_m) > FIT_STAGNATION_DISTANCE_LIMIT_M:
        stats.time_without_progress = 0.0
        stats.last_stagnation_dist_m = dist_m
    else:
        stats.time_without_progress += dt

    # ==========================================
    # CHECK COLLISION
    # ==========================================
    if drone.check_collision(SCREEN_WIDTH, SCREEN_HEIGHT, obstacles, PPM):
        
        # 1. Obliczamy prędkość uderzenia
        crash_speed = math.hypot(drone._vel_x, drone._vel_y)
        
        # 2. Płaska kara za sam fakt rozbicia (np. 10.0)
        genome_any.fitness -= FIT_CRASH_BASE_PENALTY
        
        # 3. Dodatkowa kara za wlot w ścianę bez hamowania (np. 15.0)
        if crash_speed > SAFE_CRASH_SPEED_M_S:
            genome_any.fitness -= FIT_KAMIKAZE_PENALTY
            
        to_remove = True

    if dist_m < (TARGET_SIZE_PX / PPM):
        stats.time_without_progress = 0  # reset stagnation time

        # 1. JEDNORAZOWA NAGRODA ZA ZNALEZIENIE CELU
        if not stats.has_touched_target:
           stats.has_touched_target = True
           genome_any.fitness += FIT_DISCOVERY_BONUS 

        stats.hover_time += dt

        # 2. PUNKTOWANIE HOVEROWANIA 
        if stats.hover_time > stats.max_hover_time_achieved:
            # Obliczamy tylko ten nowy, niepunktowany wcześniej ułamek sekundy
            new_time_earned = stats.hover_time - stats.max_hover_time_achieved
            
            # Nagroda rośnie z czasem zawisu, ale tylko za "nowe" sekundy
            genome_any.fitness += (
                new_time_earned * FIT_HOVER_REWARD * (1 + stats.hover_time * 10)
            )
            # Aktualizujemy rekord życiowy drona
            stats.max_hover_time_achieved = stats.hover_time

        # 3. PEŁNY SUKCES (Ukończenie poziomu)
        if stats.hover_time >= HOVER_REQUIRED_SEC:
            genome_any.fitness += FIT_HOVER_SUCCESS_REWARD
            success = True
            to_remove = True

    else:
        stats.hover_time = 0

    if genome_any.fitness <= 0.1:
        genome_any.fitness = 0.1
    
    if stats.time_without_progress > STAGNATION_LIMIT_SEC:
        to_remove = True 

    return success, to_remove

def step_training_drone(
    #current_frame: int,
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
    use_cascade: bool,
) -> tuple[bool, bool]:
    #current_time = current_frame * dt
    to_remove = False

    # get inpputs from drone sensors and internal states
    state_inputs = drone.get_inputs(
        target_pos_m=target_m,
        screen_width_px=SCREEN_WIDTH,
        screen_height_px=SCREEN_HEIGHT,
        obstacles=obstacles,
        PPM=PPM,
        use_cascade=use_cascade,
    )

    net_action = net.activate(state_inputs)

    #TODO - wybór eksperta dla trybu cascade lub raw (albo przeliczenie wyjścia eksperta na odpowiedni format)
    #ew porównywać po przeliczeniu thrustów drona z wektora
    # ==========================================
    # 3. IMITATION LEARNING (Porównanie z Ekspertem)
    # ==========================================

    if use_cascade:
        l_thrust, r_thrust = global_flight_controller.get_motor_thrusts(
            drone=drone,
            target_x=net_action[0],
            target_y=net_action[1]
        )
        drone.set_engine_thrust(l_thrust, r_thrust)
    else:
        drone.set_engine_thrust(net_action[0], net_action[1])

    drone.update(dt)  

    return apply_fitness_rules(
        drone=drone,
        stats=stats,
        genome=genome,
        target_m=target_m,
        dt=dt,
        obstacles=obstacles,
        difficulty_multiplier=difficulty_multiplier,
        SCREEN_WIDTH=SCREEN_WIDTH,
        SCREEN_HEIGHT=SCREEN_HEIGHT,
        PPM=PPM
    )


def _eval_genome_headless(genome: neat.DefaultGenome, config: neat.Config) -> float:
    """Samotna symulacja jednego drona dla pojedynczego rdzenia procesora."""
    is_cascade = (config.genome_config.num_inputs == 16)
    expert = HardcodedBrain()

    # Losowanie mapy
    start_px, target_px = generate_start_and_target(
        SCREEN_WIDTH, SCREEN_HEIGHT, MAP_MARGIN_PX, MIN_SPAWN_DIST_M
    )
    target_m = (target_px[0] / PPM, target_px[1] / PPM)
    
    # Przeszkody na 0 do momentu wdrożenia curriculum
    obstacles = generate_obstacles(start_px, target_px, num_obstacles=0) 

    net, drone, stats = _prepare_drone_and_stats(genome, config, start_px, target_px, PPM)

    max_frames = FPS * SIMULATION_TIME
    dt = 1.0 / FPS
    current_frame = 0

    # Główna pętla logiczna - kręci się tak szybko, jak pozwala procesor
    while current_frame < max_frames:
        current_frame += 1
        
        success, should_remove = step_training_drone(
            #current_frame=current_frame,
            dt=dt,
            drone=drone,
            target_m=target_m,
            stats=stats,
            genome=genome,
            net=net,
            expert=expert,
            help_weight=0.0,
            obstacles=obstacles,
            difficulty_multiplier=1.0,
            use_cascade=is_cascade,
        )
        if should_remove:
            break

    return cast(Any, genome).fitness


def _eval_genomes_visual(genomes: list[tuple[int, neat.DefaultGenome]], config: neat.Config) -> None:
    global generation_count
    global render_graphics
    global target_fps
    global uncapped
    screen = pygame.display.get_surface()
    clock = pygame.time.Clock()

    expert = HardcodedBrain()

    is_cascade = (config.genome_config.num_inputs == 16)

    # do poprawy -> nie liczba rund, tylko jeżeli osiągnie dany fitness to obniża pomoc eksperta
    max_help_gens = 50
    help_weight = max(0, 1.0 - (generation_count / max_help_gens))

    #do przemyślenia zachowanie eksperta, oceny fitnessu po przejściu do trudniejszych scenariuszy

    for genome_id, genome in genomes:
        cast(
            Any, genome
        ).fitness = FIT_START_CAPITAL  # lub np. 0.0, jeśli użyjesz akumulacji

    # 2. Definiujemy nasze 3 rundy (Test Suite)
    scenarios: list[tuple[str, int]] = [
        ("Runda 1: Otwarte Niebo", 0),
        # ("Runda 2: Standard", 3),
        # ("Runda 3: Tor Przeszkód", 4),
    ]
    
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
            SCREEN_WIDTH, SCREEN_HEIGHT, MAP_MARGIN_PX, MIN_SPAWN_DIST_M
        )
        target_m: tuple[float, float] = (target_px[0] / PPM, target_px[1] / PPM)
        obstacles = generate_obstacles(start_px, target_px, num_obs)

        for _, genome in genomes:
            net, new_drone, new_stats = _prepare_drone_and_stats(
                genome, config, start_px, target_px, PPM
            )

            # 4. Dodawanie do list (kolejność musi być identyczna we wszystkich listach!)
            nets.append(net)
            drones.append(new_drone)
            stats_list.append(new_stats)
            ge.append(genome)

        max_frames = FPS * SIMULATION_TIME
        dt = 1.0 / FPS
        current_frame = 0

        while current_frame < max_frames and drones:
            current_frame += 1
            
            # 1. ZARZĄDZANIE CZASEM
            if not uncapped:
                clock.tick(target_fps)
            else:
                clock.tick() # Odpychanie okna, brak limitu
                
            # 2. OBSŁUGA ZDARZEŃ (W locie)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Wł/Wył renderowanie
                        render_graphics = not render_graphics
                    if event.key == pygame.K_u:  # Wł/Wył limit FPS
                        uncapped = not uncapped
                    if event.key == pygame.K_1:  # Bardzo wolno (Debug)
                        target_fps = 5
                    if event.key == pygame.K_2:  # Normalnie
                        target_fps = 60

            # 3. CZYSTA LOGIKA (Dla każdego drona)
            to_remove = []
            for i, drone in enumerate(drones):
                success, should_remove = step_training_drone(
                    #current_frame=current_frame,
                    dt=dt,
                    drone=drone,
                    target_m=target_m,
                    stats=stats_list[i],
                    genome=ge[i],
                    net=nets[i],
                    expert=expert,
                    help_weight=help_weight,
                    obstacles=obstacles,
                    difficulty_multiplier=1.0,
                    use_cascade=is_cascade,
                )
                if should_remove:
                    to_remove.append(i)

            for index in reversed(to_remove):
                _remove_drone(index, drones, stats_list, nets, ge)

            # 4. RENDEROWANIE ODPINANE
            if render_graphics:
                render_simulation(screen, drones, target_px, obstacles, PPM)
                # Możesz dodać proste info na ekranie:
                # font.render(f"FPS: {int(clock.get_fps())} | Render: {render_graphics}", ...)
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
# TRYB VISUAL (Z Okienkiem Pygame)
# =====================================================================

def run_neat_visual(config_path: str, checkpoint: str | None = None, use_cascade: bool = True) -> None:
    """Uruchamia ewolucję w 1 wątku z możliwością renderowania (Pygame)."""
    global USE_FLIGHT_CONTROLLER
    USE_FLIGHT_CONTROLLER = use_cascade
    # 1. Setup okna Pygame
    pygame.init()
    pygame.font.init()
    pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("BIAI Drone Sim - AI Evolution (VISUAL)")

    # 2. Pobranie gotowej populacji z naszej funkcji pomocniczej
    population, _ = _setup_population(config_path, checkpoint)

    print("Rozpoczynanie ewolucji w trybie WIZUALNYM...")
    
    # Uruchamiamy eval_genomes_visual (Twój zrefaktoryzowany kod z poprzednich kroków)
    winner = population.run(_eval_genomes_visual, EVOLUTION_CYCLES)

    print(f"\nBest genome found:\n{winner}")
    model_path = "models/best_drone_cascade.pkl" if USE_FLIGHT_CONTROLLER else "models/best_drone_e2e.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(winner, f)
        print(f"Zapisano najlepszego drona do '{model_path}'")

    pygame.quit()


# =====================================================================
# TRYB HEADLESS (Wieloprocesowy, Bez okienka)
# =====================================================================

def run_neat_headless(config_path: str, checkpoint: str | None = None, use_cascade: bool = True) -> None:
    """Uruchamia ewolucję na wszystkich rdzeniach procesora bez GUI."""
    # UWAGA: Zero importów i initów Pygame tutaj!
    global USE_FLIGHT_CONTROLLER
    USE_FLIGHT_CONTROLLER = use_cascade
    
    population, _ = _setup_population(config_path, checkpoint)

    # Użycie wszystkich dostępnych rdzeni procesora, 1 wolny
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Rozpoczynanie ewolucji w trybie HEADLESS (Używam {num_cores} rdzeni)...")
    
    # Tworzymy Parallel Evaluator podając mu naszą zrefaktoryzowaną funkcję dla 1 drona
    pe = neat.ParallelEvaluator(num_cores, _eval_genome_headless)
    
    winner = population.run(pe.evaluate, EVOLUTION_CYCLES)

    print(f"\nBest genome found:\n{winner}")
    model_path = "models/best_drone_cascade.pkl" if USE_FLIGHT_CONTROLLER else "models/best_drone_e2e.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(winner, f)
        print(f"Zapisano najlepszego drona do '{model_path}'")