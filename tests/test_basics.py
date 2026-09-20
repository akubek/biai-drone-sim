import random

import neat
import pytest

from src.config.config import (
    GRID_SIZE_M,
    PPM,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)
from src.core.drone import Drone
from src.core.map_generator import generate_grid_obstacles


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