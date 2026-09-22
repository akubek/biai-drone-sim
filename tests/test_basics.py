import math
import random

import neat
import pytest

from src.ai.fitness import compute_fitness
from src.ai.neat_eval import check_termination
from src.config.config import (
    GRID_SIZE_M,
    PPM,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)
from src.core.drone import Drone
from src.core.environment import TIERS, generate_scenarios
from src.core.map_generator import generate_grid_obstacles
from src.core.stats import EndReason, EvolutionStats


def test_hover_thrust_keeps_altitude():
    """Continuous thrust counteracting gravity on both engines = no descent."""
    drone = Drone(2.0, 1.5)
    hover = (drone.mass * drone.gravity) / (2 * drone.max_thrust)
    y_start = drone._y

    drone.set_engine_thrust(hover, hover)
    for _ in range(300):                       # 5 sekund przy 60 Hz
        drone.update(1.0 / 60.0)

    # Zmierzone przed Milestone 1: dryf 0.152 m, v_y -> 0 po ok. 5 s.
    # Blad startowy (silniki rozpedzaja sie od zera), nie ciagly dryf - swiadomie nie naprawiany.
    assert abs(drone._y - y_start) < 0.20, f"drift {drone._y - y_start:.3f} m"
    assert abs(drone._vel_y) < 0.01, f"velocity in steady state (after 5s) {drone._vel_y:.4f} m/s"


def test_obstacle_count_is_exact():
    for seed in range(20):
        random.seed(seed)
        for requested in (1, 3, 5, 8):
            obstacles = generate_grid_obstacles(
                SCREEN_WIDTH, SCREEN_HEIGHT, (200, 600), (800, 150),
                GRID_SIZE_M, requested, PPM,
            )
            assert len(obstacles) == requested, \
                f"seed={seed}, requested {requested}, got {len(obstacles)}"


@pytest.mark.parametrize("use_cascade,expected", [(True, 16), (False, 18)])
def test_input_vector_shape_and_range(use_cascade, expected):
    """The number of inputs must match conf/neat-*.txt, values are normalized."""
    drone = Drone(2.0, 1.5)
    inputs = drone.get_inputs(
        target_pos_m=(3.0, 1.0),
        screen_width_px=SCREEN_WIDTH, screen_height_px=SCREEN_HEIGHT,
        obstacles=[], PPM=PPM, use_cascade=use_cascade,
    )
    assert len(inputs) == expected
    assert all(-1.0 <= v <= 1.0 for v in inputs), \
        f"out of range: {[v for v in inputs if not -1.0 <= v <= 1.0]}"


def test_collision_detected_at_screen_edge():
    """Drone outside the map must be detected as a collision, inside - not."""
    inside = Drone(SCREEN_WIDTH / PPM / 2, SCREEN_HEIGHT / PPM / 2)
    assert not inside.check_collision(SCREEN_WIDTH, SCREEN_HEIGHT, [], PPM)

    outside = Drone(0.05, SCREEN_HEIGHT / PPM / 2)
    assert outside.check_collision(SCREEN_WIDTH, SCREEN_HEIGHT, [], PPM)

def test_hover_drift_report(capsys):
    """Drift measurement in steady state - does static hover thrust suffice?

    This is not a pass/fail test, but a measurement for decision making in #9b.
    Run with -s to see the numbers.
    """
    drone = Drone(2.0, 1.5)
    hover = (drone.mass * drone.gravity) / (2 * drone.max_thrust)
    y_start = drone._y
    dt = 1.0 / 60.0

    drone.set_engine_thrust(hover, hover)

    checkpoints = {}
    for step in range(1, 601):              # 10 sekund
        drone.update(dt)
        t = step * dt
        if step in (30, 60, 120, 300, 600):  # 0.5s, 1s, 2s, 5s, 10s
            checkpoints[t] = (drone._y - y_start, drone._vel_y)

    with capsys.disabled():
        print("\n--- Drift with static hover thrust ---")
        for t, (drift, vel) in checkpoints.items():
            print(f"  t={t:5.1f}s   drift={drift:+.4f} m   v_y={vel:+.4f} m/s")

    # Assertion only for obviously broken cases (wrong sign, no gravity, wrong units).
    assert abs(drone._y - y_start) < 1.0, "drift above one meter - something is fundamentally wrong"

@pytest.mark.parametrize("conf", ["conf/neat-cascade.txt", "conf/neat-e2e.txt"])
def test_network_is_not_dead(conf):
    """Network must react to inputs. A mismatch between feed_forward and the builder results in all zeros."""
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, conf)
    pop = neat.Population(config)
    genome = next(iter(pop.population.values()))
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    n = config.genome_config.num_inputs
    out_a = net.activate([0.5] * n)
    out_b = net.activate([-0.9, 0.8] + [0.1] * (n - 2))

    assert any(v != 0.0 for v in out_a), "network returns all zeros - check feed_forward"
    assert out_a != out_b, "network does not react to input change"


@pytest.mark.parametrize("conf", ["conf/neat-cascade.txt", "conf/neat-e2e.txt"])
def test_genomes_share_node_keys(conf):
    """Genomes must share node keys, otherwise distance and crossover are meaningless."""
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, conf)
    pop = neat.Population(config)
    g0, g1 = list(pop.population.values())[:2]

    shared = set(g0.connections) & set(g1.connections)
    assert len(shared) == len(g0.connections), \
        f"only {len(shared)}/{len(g0.connections)} shared connections - num_hidden > 0?"
    assert g0.distance(g1, config.genome_config) < 1.0, "initial genomes are too distant"

def test_crash_returns_crash_reason():
    """Drone spawned in a wall must return EndReason.CRASH."""
    drone = Drone(0.1, 1.0)          # tuz przy lewej krawedzi
    stats = EvolutionStats(initial_dist_m=2.0, min_dist_m=2.0,
                           last_stagnation_dist_m=2.0,
                           max_allowed_escape_dist_m=10.0)

    reason = check_termination(drone, stats, (2.0, 1.0), 1 / 60, [])
    assert reason is EndReason.CRASH


def test_normal_flight_returns_none():
    """Drone in the middle of the map, without collisions, should continue flying."""
    drone = Drone(2.5, 1.8)
    stats = EvolutionStats(initial_dist_m=1.0, min_dist_m=1.0,
                           last_stagnation_dist_m=1.0,
                           max_allowed_escape_dist_m=10.0)


    assert check_termination(drone, stats, (3.0, 1.8), 1 / 60, []) is None

def test_crash_scores_lower_than_success():
    stats = EvolutionStats(initial_dist_m=3.0, min_dist_m=0.1,
                           max_hover_time_achieved=1.5, has_touched_target=True)
    assert (compute_fitness(stats, EndReason.SUCCESS).total
            > compute_fitness(stats, EndReason.CRASH).total)

def test_progress_is_scale_free():
    """The same fraction of the closed distance = the same result, regardless of the map scale."""
    blisko = EvolutionStats(initial_dist_m=1.0, min_dist_m=0.5)
    daleko = EvolutionStats(initial_dist_m=4.0, min_dist_m=2.0)
    assert (compute_fitness(blisko, EndReason.TIMEOUT).progress
            == compute_fitness(daleko, EndReason.TIMEOUT).progress)

def test_fast_pass_through_target_is_not_hover():
    """Flying through the target zone at high speed should not accumulate hover time."""
    drone = Drone(2.5, 1.8)
    drone._vel_x = 2.0                      # duzo powyzej progu
    stats = EvolutionStats(initial_dist_m=1.0, min_dist_m=1.0,
                           last_stagnation_dist_m=1.0,
                           max_allowed_escape_dist_m=10.0)

    for _ in range(120):                    # 2 s w strefie celu
        check_termination(drone, stats, (2.5, 1.8), 1 / 60, [])
        drone._vel_x = 2.0                  # utrzymuj predkosc

    assert stats.hover_time == 0.0
    assert stats.has_touched_target is True

def test_hovering_throttle_matches_base_hover():
    """Drone maintaining a hover should consume thrust equal to the base hover value."""
    drone = Drone(2.5, 1.8)
    hover = (drone.mass * drone.gravity) / (2 * drone.max_thrust)
    stats = EvolutionStats(initial_dist_m=1.0, min_dist_m=1.0,
                           last_stagnation_dist_m=1.0,
                           max_allowed_escape_dist_m=10.0)
    drone.set_engine_thrust(hover, hover)
    dt = 1 / 60
    for _ in range(300):
        drone.update(dt)
        stats.energy_raw += (drone.actual_l_thrust + drone.actual_r_thrust) * dt
        stats.total_time_alive += dt

    throttle = stats.energy_raw / (2.0 * stats.total_time_alive)
    assert abs(throttle - hover) < 0.01, f"throttle {throttle:.4f} vs hover {hover:.4f}"

def test_tier_controls_obstacle_count():
    """Parametr tier musi docierac do generatora, nie byc po cichu ignorowany."""
    for tier, spec in TIERS.items():
        scenarios = generate_scenarios(count=5, tier=tier)
        assert all(len(s.obstacles_px) == spec["obstacles"] for s in scenarios)
        assert all(s.tier == tier for s in scenarios)


def test_tier_distance_band():
    """Dystans start-cel musi miescic sie w pasmie zdefiniowanym dla poziomu."""
    for tier, spec in TIERS.items():
        lo, hi = spec["dist_m"]
        for s in generate_scenarios(count=10, tier=tier):
            d = math.hypot(s.target_px[0] - s.start_px[0],
                           s.target_px[1] - s.start_px[1]) / PPM
            assert lo <= d <= hi, f"tier {tier}: {d:.2f} m poza {lo}-{hi}"