import math
from typing import Any

from src.core.drone import Drone
# Zależnie od tego, czy już przeniosłeś pliki, importuj z src.core.controller:
from src.core.flight_controller import FlightController 


class HardcodedBrain:
    """Vector algorithm navigation expert."""

    def __init__(self):
        pass

    def get_target_commands(self, drone: Drone, target_m: tuple[float, float]) -> tuple[float, float]:
        """Returns [desired_angle_in_radians, desired_thrust_in_percent]."""
        dx = target_m[0] - drone._x
        dy = target_m[1] - drone._y

        dist_to_target = math.hypot(dx, dy)

        # =========================================================
        # 1. AVOIDANCE: CALCULATE REPULSIVE FORCE FROM SENSORS
        # =========================================================
        repulsive_x = 0.0
        repulsive_y = 0.0

        drone_radius = drone.width_m / 2.0
        base_safe_dist = drone_radius * 1.5
        hard_limit = drone_radius * 1.1
        #max_emergency_tilt = 0.6
        max_lateral_force = (2 * drone.max_thrust) * 0.2
        max_lateral_accel = max_lateral_force / drone.mass

        avoid_gain = 1.0  
        brake_gain = 0.4  

        for i, dist in enumerate(drone.last_sensor_data):
            rad = drone._angle + drone.sensor_angles[i]
            dir_x = math.sin(rad)
            dir_y = -math.cos(rad)

            approach_speed = (drone._vel_x * dir_x) + (drone._vel_y * dir_y)

            braking_dist = 0.0
            if approach_speed > 0:
                braking_dist = (approach_speed**2) / (2 * max_lateral_accel)

            dynamic_safe_dist = base_safe_dist + braking_dist

            if dist < dynamic_safe_dist:
                dist_strength = (dynamic_safe_dist - dist) * avoid_gain
                speed_strength = max(0.0, approach_speed) * brake_gain

                raw_strength = dist_strength + speed_strength

                parking_zone_m = 1.0
                dist_factor = min(1.0, dist_to_target / parking_zone_m)

                current_speed = math.hypot(drone._vel_x, drone._vel_y)
                speed_factor = min(1.0, current_speed / 1.5)

                goal_factor = max(dist_factor, speed_factor)

                if dist < base_safe_dist:
                    panic_factor = 1.0 - (dist - hard_limit) / (base_safe_dist - hard_limit)
                    panic_factor = max(0.0, min(1.0, panic_factor))
                else:
                    panic_factor = 0.0

                final_factor = max(goal_factor, panic_factor)
                strength = raw_strength * final_factor

                repulsive_x -= dir_x * strength
                repulsive_y += dir_y * strength

        # =========================================================
        # 2. POSITION PD CONTROLLER (desired joystick vector)
        # =========================================================
        p_gain = 0.16
        d_gain = 0.08

        # --- OŚ X (Pochylenie / Ruch na boki) ---
        target_x = (dx * p_gain) - (drone._vel_x * d_gain) + repulsive_x
        target_x = max(-1.0, min(1.0, target_x))

        # --- OŚ Y (Wznoszenie / Opadanie) ---
        # dy jest dodatnie gdy cel jest pod nami (chcemy opadać -> target_y > 0)
        # dy jest ujemne gdy cel jest nad nami (chcemy się wznosić -> target_y < 0)
        # Odejmujemy repulsive_y, bo repulsive_y to wektor odpychający
        target_y = (dy * p_gain) - (drone._vel_y * d_gain) - repulsive_y
        target_y = max(-1.0, min(1.0, target_y))

        return target_x, target_y

    def activate(self, drone: Drone, target_m: tuple[float, float]) -> tuple[float, float]:
        """Maintain old interface for compatibility."""
        return self.get_target_commands(drone, target_m)