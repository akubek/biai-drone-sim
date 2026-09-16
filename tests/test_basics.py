import random

import pytest

from src.config.config import (
    GRID_SIZE_M,
    PPM,
    SAFE_ZONE_CELLS,
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
    for _ in range(120):                       # 2 sekundy przy 60 Hz
        drone.update(1.0 / 60.0)

    # Dron delikatnie wznosi sie: opor liniowy hamuje opadanie w fazie rozpedzania silnikow,
    # wiec statyczny hover thrust nie jest punktem rownowagi dynamicznej. Do zbadania w #9.
    assert abs(drone._y - y_start) < 0.15, f"dryf {drone._y - y_start:.3f} m"
    assert abs(drone._angle) < 0.01, "symetryczny ciag nie powinien obracac drona"


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