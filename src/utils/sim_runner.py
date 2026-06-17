import os
import sys
import math
import pickle
import pygame
from typing import cast, Any
import neat

from src.core.drone import Drone
from src.core.flight_controller import FlightController
from src.ai.expert import HardcodedBrain
from src.core.stats import EvolutionStats
from src.core.environment import generate_obstacles, generate_start_and_target
from src.utils.renderer import render_simulation
from src.ai.neat_eval import apply_fitness_rules
from src.config.config import *
from src.config.rewards import *
from src.config.evolution import *


def reset_test_drone(target_m: tuple[float, float]) -> tuple[Drone, EvolutionStats, Any]:
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

    class DummyGenome:
        fitness = FIT_START_CAPITAL

    return new_drone, new_stats, DummyGenome()


def test_best_drone(config_path: str, genome_path: str = "best_drone.pkl") -> None:
    """Wczytuje najlepszego drona z pliku i pozwala go przetestować."""
    
    if not os.path.exists(genome_path):
        print(f"❌ BŁĄD: Nie znaleziono zapisanego modelu '{genome_path}'.")
        return

    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path
    )
    
    with open(genome_path, "rb") as f:
        winner_genome = pickle.load(f)

    # Dedukcja trybu architektury z konfiguracji
    is_cascade = (config.genome_config.num_inputs == 16)
    flight_controller = FlightController() if is_cascade else None

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"BIAI Drone Sim - CHAMPION ({'CASCADE' if is_cascade else 'E2E'})")
    clock = pygame.time.Clock()

    net = neat.nn.FeedForwardNetwork.create(winner_genome, config)
    drone = Drone((SCREEN_WIDTH // 2) / PPM, (SCREEN_HEIGHT // 2) / PPM)

    target_pos = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4)
    drone_pos_px: tuple[int, int] = cast(tuple[int, int], (int(drone._x * PPM), int(drone._y * PPM)))
    obstacles = generate_obstacles(drone_pos_px, target_pos, num_obstacles=2)

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                drone_pos_px = cast(tuple[int, int], (int(drone._x * PPM), int(drone._y * PPM)))
                obstacles = generate_obstacles(drone_pos_px, target_pos, num_obstacles=2)

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
            print("--- KONIEC PRÓBY: Kolizja ---")
            drone, stats, genome = reset_test_drone(target_m)
            continue

        render_simulation(screen, [drone], target_px, obstacles, PPM)
        clock.tick(FPS)

    pygame.quit()


def test_baseline() -> None:
    """Testuje działanie HardcodedBrain (Eksperta)."""
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
    target_m = (target_px[0] / PPM, target_px[1] / PPM)
    obstacles = []

    drone, stats, genome = reset_test_drone(target_m)

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
                drone_pos_px = (int(drone._x * PPM), int(drone._y * PPM))
                obstacles = generate_obstacles(drone_pos_px, target_px, num_obstacles=5)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                target_px = pygame.mouse.get_pos()
                target_m = (target_px[0] / PPM, target_px[1] / PPM)
                # Szybki reset statystyk by Ekspert dostał poprawny cel
                d_curr = math.hypot(target_m[0] - drone._x, target_m[1] - drone._y)
                drone, stats, genome = reset_test_drone(target_m)

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
        success, should_stop = apply_fitness_rules(
            drone=drone,
            stats=stats,
            genome=genome,
            target_m=target_m,
            dt=dt,
            obstacles=obstacles,
            difficulty_multiplier=1.0,
            SCREEN_WIDTH=SCREEN_WIDTH,
            SCREEN_HEIGHT=SCREEN_HEIGHT,
            PPM=PPM
        )

        if should_stop or success or frames >= max_frames:
            status = "SUKCES" if success else "KOLIZJA/EWALUACJA" if should_stop else "TIMEOUT"
            print(f"--- KONIEC PRÓBY: {status} | Czas: {current_time_sec:.1f}s | Punkty: {genome.fitness:.1f} ---")

            drone, stats, genome = reset_test_drone(target_m)
            frames = 0

        # WIZUALIZACJA
        render_simulation(screen, [drone], target_px, obstacles, PPM)

        txt_dist = font.render(f"Dystans: {dist_m:.2f} m", True, (0, 255, 255))
        txt_time = font.render(f"Czas: {current_time_sec:.1f} s", True, (255, 255, 255))
        screen.blit(txt_dist, (10, 10))
        screen.blit(txt_time, (10, 40))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()