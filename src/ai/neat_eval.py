import math
import multiprocessing
import pickle
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, cast

import neat
import pygame
from neat.nn import FeedForwardNetwork

from src.ai.curriculum import CurriculumController
from src.ai.evaluator import CurriculumParallelEvaluator
from src.ai.expert import HardcodedBrain
from src.ai.fitness import compute_fitness, hover_credit, ramp
from src.ai.holdout import HoldoutReporter
from src.ai.state import TrainingState
from src.config.config import *
from src.config.evolution import *
from src.config.physics import *
from src.config.rewards import *
from src.core.drone import Drone
from src.core.environment import Scenario, load_holdout
from src.core.flight_controller import FlightController
from src.core.stats import EndReason, EpisodeResult, EvolutionStats
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


def check_termination(
        drone: Drone, 
        stats: EvolutionStats,
        target_m: tuple[float, float], 
        dt: float, 
        obstacles: list, 
        SCREEN_WIDTH: int = SCREEN_WIDTH,
        SCREEN_HEIGHT: int = SCREEN_HEIGHT,
        PPM: float = PPM
    ) -> EndReason | None:
    """Calculates fitness and returns the reason for the end of the episode or none if the episode (flight) is still ongoing."""
    dist_m = math.hypot(drone._x - target_m[0], drone._y - target_m[1])

    # escape early check
    if dist_m > stats.max_allowed_escape_dist_m:
        return EndReason.ESCAPE

    # spinout check
    if abs(drone._angular_vel) > MAX_SAFE_ANGULAR_VEL:
        stats.spinout_time += dt
        if stats.spinout_time > MAX_ALLOWED_SPINOUT_TIME:
            return EndReason.SPINOUT
    else:
        stats.spinout_time = 0.0

    # stagnation check
    if (stats.last_stagnation_dist_m - dist_m) > FIT_STAGNATION_DISTANCE_LIMIT_M:
        stats.time_without_progress = 0.0
        stats.last_stagnation_dist_m = dist_m
    else:
        stats.time_without_progress += dt

    # collision check
    if drone.check_collision(SCREEN_WIDTH, SCREEN_HEIGHT, obstacles, PPM):
        stats.crash_speed = math.hypot(drone._vel_x, drone._vel_y)
        return EndReason.CRASH

    # target check
    if dist_m < (TARGET_SIZE_PX / PPM):
        stats.time_without_progress = 0.0  # reset stagnation time
        stats.has_touched_target = True

        speed = math.hypot(drone._vel_x, drone._vel_y)
        is_stable = (speed <= HOVER_MAX_SPEED_M_S
                     and abs(drone._angular_vel) <= HOVER_MAX_ANGULAR_VEL)

        if is_stable:
            stats.hover_time_s += dt
            stats.max_hover_time_achieved = max(stats.max_hover_time_achieved, stats.hover_time_s)

            # hover success check
            if stats.hover_time_s >= HOVER_REQUIRED_SEC:
                return EndReason.SUCCESS
        # reset hover time if not stable
        else:
            stats.hover_time_s = 0.0

        stats.hover_credit_s += dt * hover_credit(speed, abs(drone._angular_vel))
        stats.max_hover_credit_s = max(stats.max_hover_credit_s,
                                   stats.hover_credit_s)
        stats.lin_credit_s += dt * ramp(speed, HOVER_MAX_SPEED_M_S, V_REF)
        stats.ang_credit_s += dt * ramp(abs(drone._angular_vel), HOVER_MAX_ANGULAR_VEL, W_REF)
        stats.max_lin_credit_s = max(stats.max_lin_credit_s, stats.lin_credit_s)
        stats.max_ang_credit_s = max(stats.max_ang_credit_s, stats.ang_credit_s)
    # reset hover time if not at target
    else:
        stats.hover_time_s = 0.0
        stats.hover_credit_s = 0.0
        stats.lin_credit_s = 0.0
        stats.ang_credit_s = 0.0

    # stagnation check
    if stats.time_without_progress > STAGNATION_LIMIT_SEC:
        return EndReason.STAGNATION

    return None

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
) -> EndReason | None:
    #current_time = current_frame * dt

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
    stats.energy_raw += (drone.actual_l_thrust + drone.actual_r_thrust) * dt
    stats.total_time_alive += dt
    stats.accumulated_rotation += abs(drone._angular_vel) * dt
    dist_m = math.hypot(drone._x - target_m[0], drone._y - target_m[1]) # TODO: maybe not important - we calculate dist_m in check termination as well.
    stats.observe_distance(dist_m, math.hypot(drone._vel_x, drone._vel_y), abs(drone._angular_vel))

    return check_termination(
        drone=drone,
        stats=stats,
        target_m=target_m,
        dt=dt,
        obstacles=obstacles,
        SCREEN_WIDTH=SCREEN_WIDTH,
        SCREEN_HEIGHT=SCREEN_HEIGHT,
        PPM=PPM
    )

def _run_episode(
    genome: neat.DefaultGenome,
    config: neat.Config,
    scenario: Scenario,
    expert: HardcodedBrain,
    help_weight: float,
) -> EpisodeResult:
    """One flight of a single drone on a single scenario."""
    # Network is rebuilt for each scenario - RecurrentNetwork keeps
    # state between activate() calls, otherwise map N+1 would start
    # with the memory from map N.
    net, drone, stats = _prepare_drone_and_stats(
        genome, config, scenario.start_px, scenario.target_px, PPM
    )

    obstacles = scenario.rects()
    target_m = scenario.target_m(PPM)
    dt = 1.0 / FPS
    max_frames = FPS * SIMULATION_TIME
    use_cascade = cast(Any, config).use_cascade

    reason: EndReason | None = None
    for _ in range(max_frames):
        reason = step_training_drone(
            dt=dt, drone=drone, target_m=target_m, stats=stats,
            genome=genome, net=net, expert=expert,
            help_weight=help_weight, obstacles=obstacles,
            use_cascade=use_cascade,
        )
        if reason is not None:
            break

    if reason is None:
        reason = EndReason.TIMEOUT

    components = compute_fitness(stats, reason)
    cast(Any, genome).fitness = components.total

    return EpisodeResult.from_stats(
        stats=stats,
        components=components,
        reason=reason,
        max_episode_time_s=SIMULATION_TIME,
    )

def _eval_genome_headless(genome: neat.DefaultGenome, config: neat.Config) -> tuple[float, EpisodeResult]:
    """Single simulation of one drone for a single CPU core."""
    expert = HardcodedBrain()
    help_weight = getattr(config, 'current_help_weight', 0.0)
    scenarios: list[Scenario] = getattr(config, "shared_scenarios", [] )

    result = [
        _run_episode(genome, config, sc, expert, help_weight)
        for sc in scenarios
    ]

    return cast(Any, genome).fitness, result


def _eval_genomes_visual(genomes: list[tuple[int, neat.DefaultGenome]], config: neat.Config) -> None:
    global render_graphics, target_fps, uncapped

    screen = pygame.display.get_surface()
    clock = pygame.time.Clock()
    expert = HardcodedBrain()

    global_state.update_parameters()

    scenarios = global_state.scenarios_for_generation()

    episode_results: dict[int, list[EpisodeResult]] = {g.key: [] for _, g in genomes}
    total_population = len(genomes)
    
    for scenario in scenarios:
        nets: list[Any] = []
        ge: list[neat.DefaultGenome] = []
        drones: list[Drone] = []
        stats_list: list[EvolutionStats] = []

        target_m = scenario.target_m(PPM)
        obstacles = scenario.rects()

        for _, genome in genomes:
            net, new_drone, new_stats = _prepare_drone_and_stats(
                genome, config, scenario.start_px, scenario.target_px, PPM
            )
            nets.append(net)
            drones.append(new_drone)
            stats_list.append(new_stats)
            ge.append(genome)

        max_frames = FPS * SIMULATION_TIME
        dt = 1.0 / FPS
        current_frame = 0

        while current_frame < max_frames and drones:
            current_frame += 1
            current_time_sec = current_frame * dt

            if not uncapped:
                clock.tick(target_fps)
            else:
                clock.tick()
                
            # 2. OBSŁUGA ZDARZEŃ
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

            # 3. Logika drona
            to_remove = []
            for i, drone in enumerate(drones):
                end_reason = step_training_drone(
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

                if end_reason is not None:
                    components = compute_fitness(stats_list[i], end_reason)
                    cast(Any, ge[i]).fitness = components.total
                    episode_results[ge[i].key].append(EpisodeResult.from_stats(
                        stats=stats_list[i],
                        components=components,
                        reason=end_reason,
                        max_episode_time_s=SIMULATION_TIME
                    ))
                    to_remove.append(i)

            for index in reversed(to_remove):
                _remove_drone(index, drones, stats_list, nets, ge)

            # 4. Conditional rendering
            if render_graphics:
                closest_m = min((s.min_dist_m for s in stats_list), default=0.0)
                render_simulation(screen, drones, scenario.target_px, obstacles, PPM)
                render_neat_hud(
                    screen=screen,
                    font=font,
                    generation=global_state.generation,
                    alive_count=len(drones),
                    pop_size=total_population,
                    best_fitness=closest_m, #TODO: zmienic w render neat hud na closest distance
                    current_time_sec=current_time_sec
                )
                pygame.display.flip()

        # timeout drones that lived for the whole simulation, but did not reach the target
        for i in range(len(drones)):
            components = compute_fitness(stats_list[i], EndReason.TIMEOUT)
            cast(Any, ge[i]).fitness = components.total
            episode_results[ge[i].key].append(EpisodeResult.from_stats(
                stats=stats_list[i],
                components=components,
                reason=EndReason.TIMEOUT,
                max_episode_time_s=SIMULATION_TIME,
            ))

    # Fitness = average from K scenarios
    for _, genome in genomes:
        results = episode_results[genome.key]
        if not results:
            raise RuntimeError(f"genome {genome.key}: no episodes - check keys in episode_results")
        cast(Any, genome).fitness = statistics.fmean(r.fitness for r in results)

    all_results = [r for rs in episode_results.values() for r in rs]
    counts = Counter(r.end_reason.value for r in all_results)
    expected = total_population * len(scenarios)
    print(f"gen {global_state.generation}: {dict(counts)} | sum {len(all_results)}/{expected}")

    global_state.last_metrics = episode_results
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

    population, config = _setup_population(
        config_path,
        checkpoint,
        pop_size=exp_config.get("pop_size"),
        run_dir=str(run_dir),
        net_type=exp_config.get("net_type", "feedforward"),
        use_cascade=use_cascade,
    )

    holdout = HoldoutReporter(
        scenarios=load_holdout(),
        run_episode_fn=_run_episode,
        config=config,
        folder=str(run_dir),
        every=10,
    )
    population.add_reporter(holdout)
    population.add_reporter(CurriculumController(global_state, holdout))
    
    reporter = CSVTrainingReporter(
        global_state,
        folder=str(run_dir),
        filename="evolution_log.csv",
        run_id=run_dir.name
    )
    reporter.holdout = holdout
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
    
    population, config = _setup_population(
        config_path,
        checkpoint,
        pop_size=exp_config.get("pop_size"),
        run_dir=str(run_dir),
        use_cascade=use_cascade,
        net_type=exp_config.get("net_type", "feedforward"),
    )

    holdout = HoldoutReporter(
        scenarios=load_holdout(),
        run_episode_fn=_run_episode,
        config=config,
        folder=str(run_dir),
        every=10,
    )
    population.add_reporter(holdout)
    population.add_reporter(CurriculumController(global_state, holdout))

    reporter = CSVTrainingReporter(
        global_state,
        folder=str(run_dir),
        filename="evolution_log.csv",
        run_id=run_dir.name
    )
    reporter.holdout = holdout
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