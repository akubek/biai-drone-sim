import math
import multiprocessing
import pickle
import sys
from pathlib import Path
from typing import Any, cast

import neat
import pygame
from neat.nn import FeedForwardNetwork

from src.ai.evaluator import CurriculumParallelEvaluator
from src.ai.expert import HardcodedBrain
from src.ai.state import TrainingState
from src.config.config import *
from src.config.evolution import *
from src.config.physics import *
from src.config.rewards import *
from src.core.drone import Drone
from src.core.environment import generate_start_and_target
from src.core.flight_controller import FlightController
from src.core.map_generator import generate_grid_obstacles
from src.core.stats import EvolutionStats
from src.utils.logger import CSVTrainingReporter
from src.utils.renderer import render_neat_hud, render_simulation
from src.utils.run_manager import create_run_dir

pygame.font.init()
font = pygame.font.SysFont("arial", 10)

show_simulation = True

# ZMIENNE DO MIĘKKIEGO PRZEŁĄCZANIA (w trybie wizualnym)
render_graphics = True
target_fps = FPS
uncapped = False

global_flight_controller = FlightController()

global_state: TrainingState
NET_BUILDERS = {
    "feedforward": neat.nn.FeedForwardNetwork.create,
    "recurrent": neat.nn.RecurrentNetwork.create,
}

# =====================================================================
# METODY POMOCNICZE (ŚRODOWISKO I EWALUACJA)
# =====================================================================
def _apply_net_type(config: neat.Config, net_type: str):
    """Applies the network type to the NEAT configuration and returns the corresponding network builder."""
    if net_type not in NET_BUILDERS:
        raise ValueError(f"Unknown net_type: {net_type}")
    config.genome_config.feed_forward = (net_type == "feedforward")
    cast(Any, config).net_type = net_type
    return NET_BUILDERS[net_type]

def _setup_population(
        config_path: str,
        checkpoint: str | None,
        pop_size: int | None,
        run_dir: str | None,
        net_type: str = "feedforward",
        use_cascade: bool = True,
    ) -> tuple[neat.Population, neat.Config]:
    """Shared setup for NEAT population, including checkpoint handling and reporters."""
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path
    )

    _apply_net_type(config, net_type)
    cast(Any, config).use_cascade = use_cascade

    checkpoint_dir = Path(run_dir) / "checkpoints" if run_dir else Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if pop_size is not None:
        config.pop_size = pop_size

    # 1. Logika szukania najnowszego checkpointu ("latest")
    if checkpoint == "latest":
        checkpoints = [
            f for f in checkpoint_dir.iterdir()
            if f.is_file() and f.name.startswith("neat-checkpoint-")
        ]
        if checkpoints:
            latest_checkpoint_path = max(checkpoints, key=lambda x: int(x.name.split("-")[-1]))
            checkpoint = str(latest_checkpoint_path)
            print(f"Found latest checkpoint: {checkpoint}")
        else:
            print("The 'checkpoints' folder is empty. Starting from scratch.")
            checkpoint = None

    # 2. Tworzenie populacji
    if checkpoint is not None:
        print(f"Restoring evolution state from checkpoint: {checkpoint}")
        population = neat.Checkpointer.restore_checkpoint(checkpoint)
    else:
        print("Loading new population from scratch...")
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
) -> tuple[Any, Drone, EvolutionStats]:
    """Creates the network, the physical drone, and initializes statistics with limits."""
    
    # 1. Sieć NEAT
    genome_any = cast(Any, genome)
    genome_any.fitness = FIT_START_CAPITAL

    builder = NET_BUILDERS[cast(Any, config).net_type]
    net = builder(genome, config)

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
    nets: list[Any],
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
        SCREEN_WIDTH: int = SCREEN_WIDTH,
        SCREEN_HEIGHT: int = SCREEN_HEIGHT,
        PPM: float = PPM

        ) -> tuple[bool, bool]:
    """Calculates fitness and returns whether the drone has succeeded or crashed/stuck."""
    to_remove = False
    success = False
    dist_m = math.hypot(drone._x - target_m[0], drone._y - target_m[1])
    genome_any = cast(Any, genome)

    # escape early check
    if dist_m > stats.max_allowed_escape_dist_m:
        return False, True #(success, to_remove)

    # spinout check
    if abs(drone._angular_vel) > MAX_SAFE_ANGULAR_VEL:
        stats.spinout_time += dt
        if stats.spinout_time > MAX_ALLOWED_SPINOUT_TIME:
            return False, True #(success, to_remove)
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

        genome_any.fitness = max(0.1, genome_any.fitness)
        return False, True  # (success, to_remove)

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

    genome_any.fitness = max(0.1, genome_any.fitness)
    
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
    use_cascade: bool,
    SCREEN_WIDTH: int = SCREEN_WIDTH,
    SCREEN_HEIGHT: int = SCREEN_HEIGHT,
    PPM: float = PPM
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

    if use_cascade:
        net_left_thrust, net_right_thrust = global_flight_controller.get_motor_thrusts(
            drone=drone,
            target_x=net_action[0],
            target_y=net_action[1]
        )
    else:
        net_left_thrust, net_right_thrust = net_action[0], net_action[1]


    #get expert thrusts
    expert_left_thrust = 0.0
    expert_right_thrust = 0.0

    if help_weight > 0.0 and expert is not None:
        expert_action = expert.get_target_commands(drone, target_m)
        expert_left_thrust, expert_right_thrust = global_flight_controller.get_motor_thrusts(drone=drone, target_x=expert_action[0], target_y=expert_action[1])

    final_left_thrust = (net_left_thrust * (1.0 - help_weight)) + (expert_left_thrust * help_weight)
    final_right_thrust = (net_right_thrust * (1.0 - help_weight)) + (expert_right_thrust * help_weight)

    drone.set_engine_thrust(final_left_thrust, final_right_thrust)

    drone.update(dt)  

    return apply_fitness_rules(
        drone=drone,
        stats=stats,
        genome=genome,
        target_m=target_m,
        dt=dt,
        obstacles=obstacles,
        SCREEN_WIDTH=SCREEN_WIDTH,
        SCREEN_HEIGHT=SCREEN_HEIGHT,
        PPM=PPM
    )


def _eval_genome_headless(genome: neat.DefaultGenome, config: neat.Config) -> float:
    """Single simulation of one drone for a single CPU core."""
    expert = HardcodedBrain()

    help_weight = getattr(config, 'current_help_weight', 0.0)

    start_px = getattr(config, 'shared_start_px', (0, 0))
    target_px = getattr(config, 'shared_target_px', (0, 0))
    target_m = (target_px[0] / PPM, target_px[1] / PPM)
    shared_obstacles_data = getattr(config, 'shared_obstacles_data', [])
    obstacles = [
        pygame.Rect(x, y, w, h) for x, y, w, h in shared_obstacles_data
    ]

    net, drone, stats = _prepare_drone_and_stats(genome, config, start_px, target_px, PPM)

    max_frames = FPS * SIMULATION_TIME
    dt = 1.0 / FPS
    current_frame = 0
    use_cascade = cast(Any, config).use_cascade

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
            help_weight=help_weight,
            obstacles=obstacles,
            use_cascade=use_cascade,
        )
        if should_remove:
            break

        # save the success state in the genome for later analysis
        if success:
            cast(Any, genome).is_success = True

    return cast(Any, genome).fitness


def _eval_genomes_visual(genomes: list[tuple[int, neat.DefaultGenome]], config: neat.Config) -> None:
    global render_graphics
    global target_fps
    global uncapped
    global font

    screen = pygame.display.get_surface()
    clock = pygame.time.Clock()

    expert = HardcodedBrain()

    global_state.update_parameters()

    # TODO: Consider expert behavior and fitness evaluation after moving to more challenging scenarios

    for genome_id, genome in genomes:
        cast(Any, genome).fitness = FIT_START_CAPITAL

    # 2. Definiujemy nasze 3 rundy (Test Suite)
    scenarios: list[tuple[str, int]] = [
        ("Round 1: Open Sky", 0),
        # ("Round 2: Standard", 3),
        # ("Round 3: Obstacle Course", 4),
    ]

    total_population = len(genomes)
    
    for round_name, num_obs in scenarios:
        saved_fitness = {genome_id: cast(Any, g).fitness for genome_id, g in genomes}
        nets: list[Any] = []
        ge: list[neat.DefaultGenome] = []
        drones: list[Drone] = []
        stats_list: list[EvolutionStats] = []

        # 'expert' drone that already knows how to fly
        # Setup środowiska
        start_px, target_px = generate_start_and_target(
            SCREEN_WIDTH, SCREEN_HEIGHT, MAP_MARGIN_PX, MIN_SPAWN_DIST_M
        )
        target_m: tuple[float, float] = (target_px[0] / PPM, target_px[1] / PPM)
        obstacles = generate_grid_obstacles(
            SCREEN_WIDTH, SCREEN_HEIGHT,
            start_px, target_px,
            GRID_SIZE_M, global_state.num_obstacles,
            PPM
        )

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
        max_best_fitness = 0.0

        while current_frame < max_frames and drones:
            current_frame += 1
            current_time_sec = current_frame * dt

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
                    help_weight=global_state.current_help_weight,
                    obstacles=obstacles,
                    use_cascade=cast(Any, config).use_cascade,
                    SCREEN_WIDTH=SCREEN_WIDTH,
                    SCREEN_HEIGHT=SCREEN_HEIGHT,
                    PPM=PPM
                )

                # save the success state in the genome for later analysis
                if success:
                    cast(Any, ge[i]).is_success = True

                if should_remove:
                    to_remove.append(i)

            for index in reversed(to_remove):
                _remove_drone(index, drones, stats_list, nets, ge)

            # 4. RENDEROWANIE ODPINANE
            if render_graphics:
                current_best_fitness = max([cast(Any, g).fitness for g in ge]) if ge else 0.0
                max_best_fitness = max(max_best_fitness, current_best_fitness)
                render_simulation(screen, drones, target_px, obstacles, PPM)
                render_neat_hud(
                    screen=screen,
                    font=font,
                    generation=global_state.generation,
                    alive_count=len(drones),
                    pop_size=total_population,
                    best_fitness=max_best_fitness,
                    current_time_sec=current_time_sec
                )
                # Możesz dodać proste info na ekranie:
                # font.render(f"FPS: {int(clock.get_fps())} | Render: {render_graphics}", ...)
                pygame.display.flip()
        
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

    global_state.generation += 1


# =====================================================================
# VISUAL MODE (With Pygame Window)
# =====================================================================

def run_neat_visual(        
    config_path: str,
    checkpoint: str | None = None,
    use_cascade: bool = True,
    exp_config: dict | None = None
) -> None:
    """Runs the NEAT evolution in visual mode with a Pygame window."""
    global global_state
    global_state = TrainingState(exp_config=exp_config)
    # Setup the Pygame window
    pygame.init()
    pygame.font.init()
    pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("BIAI Drone Sim - AI Evolution (VISUAL)")

    exp_config = exp_config or {}
    generations = exp_config.get("generations", EVOLUTION_CYCLES)
    run_dir = create_run_dir("cascade" if use_cascade else "e2e", exp_config, config_path)
    print(f"RUN DIR: {run_dir}")

    population, _ = _setup_population(
        config_path,
        checkpoint,
        pop_size=exp_config.get("pop_size"),
        run_dir=str(run_dir),
        net_type=exp_config.get("net_type", "feedforward"),
        use_cascade=use_cascade,
    )

    reporter = CSVTrainingReporter(global_state,folder=str(run_dir), filename="evolution_log.csv")
    population.add_reporter(reporter)


    print("Starting evolution in VISUAL mode...")
    
    winner = population.run(_eval_genomes_visual, generations)

    # Save the best genome and quit the pygame window
    print(f"\nBest genome found:\n{winner}")
    with open(run_dir / "best_drone.pkl", "wb") as f:
        pickle.dump(winner, f)
        print(f"Saved best drone to '{run_dir / 'best_drone.pkl'}'")

    pygame.quit()


# =====================================================================
# HEADLESS MODE (Multiprocessing, No GUI)
# =====================================================================

def run_neat_headless(
    config_path: str,
    checkpoint: str | None = None,
    use_cascade: bool = True,
    exp_config: dict | None = None
) -> None:
    """Runs the NEAT evolution on all CPU cores without a GUI."""
    global global_state
    global_state = TrainingState(exp_config=exp_config)

    exp_config = exp_config or {}
    generations = exp_config.get("generations", EVOLUTION_CYCLES)
    run_dir = create_run_dir("cascade" if use_cascade else "e2e", exp_config, config_path)
    print(f"RUN DIR: {run_dir}")
    
    population, _ = _setup_population(
        config_path,
        checkpoint,
        pop_size=exp_config.get("pop_size"),
        run_dir=str(run_dir),
        use_cascade=use_cascade,
        net_type=exp_config.get("net_type", "feedforward"),
    )

    reporter = CSVTrainingReporter(global_state,folder=str(run_dir), filename="evolution_log.csv")
    population.add_reporter(reporter)

    # Use all available CPU cores, leaving 1 free
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Starting evolution in HEADLESS mode (Using {num_cores} cores)...")
    
    parallel_evaluator = CurriculumParallelEvaluator(num_cores, _eval_genome_headless, global_state)

    # Create a Parallel Evaluator using the _eval_genome_headless function for a single drone
    winner = population.run(parallel_evaluator.evaluate, generations)

    # Save the best genome
    print(f"\nBest genome found:\n{winner}")
    model_path = run_dir / "best_drone.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(winner, f)
        print(f"Saved best drone to '{model_path}'")