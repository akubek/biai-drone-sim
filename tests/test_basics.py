import random

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
    """Ciag rownowazacy grawitacje na obu silnikach = brak opadania."""
    drone = Drone(2.0, 1.5)
    hover = (drone.mass * drone.gravity) / (2 * drone.max_thrust)
    y_start = drone._y

    drone.set_engine_thrust(hover, hover)
    for _ in range(300):                       # 5 sekund przy 60 Hz
        drone.update(1.0 / 60.0)

    # Zmierzone przed Milestone 1: dryf 0.152 m, v_y -> 0 po ok. 5 s.
    # Blad startowy (silniki rozpedzaja sie od zera), nie ciagly dryf - swiadomie nie naprawiany.
    assert abs(drone._y - y_start) < 0.20, f"dryf {drone._y - y_start:.3f} m"
    assert abs(drone._vel_y) < 0.01, f"predkosc w stanie ustalonym (po 5s) {drone._vel_y:.4f} m/s"


def test_obstacle_count_is_exact():
    for seed in range(20):
        random.seed(seed)
        for requested in (1, 3, 5, 8):
            obstacles = generate_grid_obstacles(
                SCREEN_WIDTH, SCREEN_HEIGHT, (200, 600), (800, 150),
                GRID_SIZE_M, requested, PPM,
            )
            assert len(obstacles) == requested, \
                f"seed={seed}, zamowiono {requested}, dostano {len(obstacles)}"


@pytest.mark.parametrize("use_cascade,expected", [(True, 16), (False, 18)])
def test_input_vector_shape_and_range(use_cascade, expected):
    """Liczba wejsc musi zgadzac sie z conf/neat-*.txt, wartosci znormalizowane."""
    drone = Drone(2.0, 1.5)
    inputs = drone.get_inputs(
        target_pos_m=(3.0, 1.0),
        screen_width_px=SCREEN_WIDTH, screen_height_px=SCREEN_HEIGHT,
        obstacles=[], PPM=PPM, use_cascade=use_cascade,
    )
    assert len(inputs) == expected
    assert all(-1.0 <= v <= 1.0 for v in inputs), \
        f"poza zakresem: {[v for v in inputs if not -1.0 <= v <= 1.0]}"


def test_collision_detected_at_screen_edge():
    """Dron poza mapa musi byc wykryty jako kolizja, w srodku - nie."""
    inside = Drone(SCREEN_WIDTH / PPM / 2, SCREEN_HEIGHT / PPM / 2)
    assert not inside.check_collision(SCREEN_WIDTH, SCREEN_HEIGHT, [], PPM)

    outside = Drone(0.05, SCREEN_HEIGHT / PPM / 2)
    assert outside.check_collision(SCREEN_WIDTH, SCREEN_HEIGHT, [], PPM)

def test_hover_drift_report(capsys):
    """Pomiar dryfu w stanie ustalonym - czy statyczny hover thrust wystarcza.

    Nie jest to test typu pass/fail, tylko pomiar do decyzji w #9b.
    Uruchom z -s, zeby zobaczyc liczby.
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
        print("\n--- Dryf przy statycznym hover thrust ---")
        for t, (drift, vel) in checkpoints.items():
            print(f"  t={t:5.1f}s   dryf={drift:+.4f} m   v_y={vel:+.4f} m/s")

    # Asercja tylko na rzeczy jawnie zepsute (zly znak, brak grawitacji, zle jednostki).
    assert abs(drone._y - y_start) < 1.0, "dryf powyzej metra - cos jest fundamentalnie nie tak"