import math
import os
import pickle
from typing import cast

import neat
import pygame

from src.ai.expert import HardcodedBrain
from src.ai.fitness import compute_fitness
from src.ai.neat_eval import check_termination
from src.config.config import *
from src.config.evolution import *
from src.config.rewards import *
from src.core.drone import Drone
from src.core.flight_controller import FlightController
from src.core.map_generator import generate_grid_obstacles
from src.core.stats import EndReason, EvolutionStats
from src.utils.renderer import render_simulation


def reset_test_drone(target_m: tuple[float, float]) -> tuple[Drone, EvolutionStats]:
    """Pomocnicza funkcja do tworzenia świeżych obiektów po resecie."""
    start_x = (SCREEN_WIDTH // 2) / PPM
    start_y = (SCREEN_HEIGHT // 2) / PPM
    new_drone = Drone(start_x, start_y)
    d_start = math.hypot(target_m[0] - new_drone._x, target_m[1] - new_drone._y)

    world_diagonal_m = math.hypot(SCREEN_WIDTH / PPM, SCREEN_HEIGHT / PPM)
    allowed_escape_dist = d_start + (world_diagonal_m * 0.6)

    new_stats = EvolutionStats(
        initial_dist_m=d_start, 
        min_dist_m=d_start,
        last_stagnation_dist_m=d_start,
        max_hover_time_achieved=0.0,
        max_allowed_escape_dist_m=allowed_escape_dist
    )

    return new_drone, new_stats


def test_best_drone(config_path: str, use_cascade: bool, genome_path: str = "best_drone.pkl") -> None:
    """Loads the best drone from a file and allows testing it."""
    
    if not os.path.exists(genome_path):
        print(f"ERROR: Could not find saved model '{genome_path}'.")
        return

    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path
    )
    
    with open(genome_path, "rb") as f:
        winner_genome = pickle.load(f)

    print("=== STRUCTURE OF THE BEST NETWORK ===")
    print(winner_genome)
    # Dedukcja trybu architektury z konfiguracji
    is_cascade = use_cascade
    flight_controller = FlightController() if is_cascade else None

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"BIAI Drone Sim - CHAMPION ({'CASCADE' if is_cascade else 'E2E'})")
    clock = pygame.time.Clock()

    net = neat.nn.FeedForwardNetwork.create(winner_genome, config)
    drone = Drone((SCREEN_WIDTH // 2) / PPM, (SCREEN_HEIGHT // 2) / PPM)

    target_pos = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4)
    drone_pos_px: tuple[int, int] = cast(tuple[int, int], (int(drone._x * PPM), int(drone._y * PPM)))
    obstacles = generate_grid_obstacles(SCREEN_WIDTH, SCREEN_HEIGHT, drone_pos_px, target_pos, GRID_SIZE_M, 5, PPM)

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                drone_pos_px = cast(tuple[int, int], (int(drone._x * PPM), int(drone._y * PPM)))
                obstacles = generate_grid_obstacles(SCREEN_WIDTH, SCREEN_HEIGHT, drone_pos_px, target_pos, GRID_SIZE_M, 5, PPM)

        # Myszka staje się nowym celem!
        mx, my = pygame.mouse.get_pos()
        target_px = (mx, my)
        target_m = (mx / PPM, my / PPM)

        inputs = drone.get_inputs(target_m, SCREEN_WIDTH, SCREEN_HEIGHT, obstacles, PPM, use_cascade=is_cascade)
        net_action = net.activate(inputs)

        if is_cascade and flight_controller is not None:
            l_thrust, r_thrust = flight_controller.get_motor_thrusts(
                drone=drone, target_x=net_action[0], target_y=net_action[1]
            )
            drone.set_engine_thrust(l_thrust, r_thrust)
        else:
            drone.set_engine_thrust(net_action[0], net_action[1])
            
        drone.update(1.0 / FPS)

        is_crashed = drone.check_collision(SCREEN_WIDTH, SCREEN_HEIGHT, obstacles, PPM)

        if is_crashed:
            print("--- END OF TEST: Collision ---")
            drone, _ = reset_test_drone(target_m)
            continue

        render_simulation(screen, [drone], target_px, obstacles, PPM)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


def test_baseline() -> None:
    """Tests the operation of the HardcodedBrain (Expert)."""
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("BIAI Drone Sim - HARDCODED BASELINE")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 24)

    expert_pilot = HardcodedBrain()
    # Baseline zawsze używa kontrolera lotu (taka jego budowa)
    flight_controller = FlightController()

    target_px = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4)
    start_x = (SCREEN_WIDTH // 2)
    start_y = (SCREEN_HEIGHT // 2)
    start_pos_px = (start_x, start_y)
    target_m = (target_px[0] / PPM, target_px[1] / PPM)
    obstacles = []

    drone, stats = reset_test_drone(target_m)

    frames = 0
    max_frames = FPS * SIMULATION_TIME 
    run = True

    while run:
        dt = 1.0 / FPS
        frames += 1
        current_time_sec = frames / FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                obstacles = generate_grid_obstacles(SCREEN_WIDTH, SCREEN_HEIGHT, start_pos_px, target_px, GRID_SIZE_M, 20, PPM)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                target_px = pygame.mouse.get_pos()
                target_m = (target_px[0] / PPM, target_px[1] / PPM)
                drone, stats = reset_test_drone(target_m)

        # 1. Sensory
        _ = drone.get_sensor_data(SCREEN_WIDTH, SCREEN_HEIGHT, obstacles, PPM)
        
        # 2. Decyzja Eksperta (zwraca wektor joysticka x, y)
        target_x, target_y = expert_pilot.activate(drone, target_m)
        
        # 3. Tłumaczenie przez Flight Controller
        l_thrust, r_thrust = flight_controller.get_motor_thrusts(
            drone=drone, target_x=target_x, target_y=target_y
        )
        drone.set_engine_thrust(l_thrust, r_thrust)

        # 4. Fizyka
        drone.update(dt)
        dist_m = math.hypot(drone._x - target_m[0], drone._y - target_m[1])

        # Ewaluacja
        reason = check_termination(
            drone=drone,
            stats=stats,
            target_m=target_m,
            dt=dt,
            obstacles=obstacles,
            SCREEN_WIDTH=SCREEN_WIDTH,
            SCREEN_HEIGHT=SCREEN_HEIGHT,
            PPM=PPM
        )

        if reason is None and frames >= max_frames:
            reason = EndReason.TIMEOUT

        if reason is not None:
            c = compute_fitness(stats, reason)
            print(f"--- END OF TEST: {reason.value:<10} | time {current_time_sec:4.1f}s "
                  f"| fitness {c.total:8.1f}  "
                  f"(progress {c.progress:6.1f}, hover {c.hover:7.1f}, "
                  f"penalties {c.crash_penalty + c.kamikaze_penalty:6.1f})")

            drone, stats = reset_test_drone(target_m)
            frames = 0

        # WIZUALIZACJA
        render_simulation(screen, [drone], target_px, obstacles, PPM)

        txt_dist = font.render(f"Distance: {dist_m:.2f} m", True, (0, 255, 255))
        txt_time = font.render(f"Time: {current_time_sec:.1f} s", True, (255, 255, 255))
        screen.blit(txt_dist, (10, 10))
        screen.blit(txt_time, (10, 40))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()