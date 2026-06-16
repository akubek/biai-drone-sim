import math
from typing import Any

from src.core.drone import Drone
# Zależnie od tego, czy już przeniosłeś pliki, importuj z src.core.controller:
from src.core.flight_controller import FlightController 


class HardcodedBrain:
    """Vector algorithm navigation expert."""

    def __init__(self):
        # Create an instance of the FlightController to handle low-level motor mixing
        self.controller = FlightController()

    def get_target_commands(self, drone: Drone, target_m: tuple[float, float]) -> tuple[float, float]:
        """Returns [desired_angle_in_radians, desired_thrust_in_percent]."""
        dx = target_m[0] - drone._x
        dy = target_m[1] - drone._y

        # =========================================================
        # 1. PHYSICS: CALCULATE IDEAL HOVER
        # =========================================================
        gravity_force = drone.mass * drone.gravity
        max_total_thrust = 2 * drone.max_thrust
        base_hover = gravity_force / max_total_thrust
        dist_to_target = math.hypot(dx, dy)

        # =========================================================
        # 2. AVOIDANCE: CALCULATE REPULSIVE FORCE FROM SENSORS
        # =========================================================
        repulsive_x = 0.0
        repulsive_y = 0.0

        drone_radius = drone.width_m / 2.0
        base_safe_dist = drone_radius * 1.5
        hard_limit = drone_radius * 1.1
        max_emergency_tilt = 0.6
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
        # 3. POSITION PD CONTROLLER + REPULSION
        # =========================================================
        p_gain = 0.16
        d_gain = 0.08

        base_pull_x = (dx * p_gain) - (drone._vel_x * d_gain)
        base_pull_x = max(-0.15, min(0.15, base_pull_x))

        lateral_push = base_pull_x + repulsive_x
        lateral_push = max(-max_emergency_tilt, min(max_emergency_tilt, lateral_push))

        base_pull_y = base_hover - (dy * p_gain) + (drone._vel_y * d_gain)
        base_pull_y = max(base_hover * 0.5, min(1.0, base_pull_y))

        upward_push = base_pull_y + repulsive_y
        upward_push = max(base_hover * 0.2, min(1.0, upward_push))

        # =========================================================
        # 4. KINEMATICS: DESIRED VeCTOR TO TARGET
        # =========================================================
        return lateral_push, -upward_push

    def activate(self, drone: Drone, target_m: tuple[float, float]) -> list[float]:
        """Maintain old interface for compatibility."""
        target_x, target_y = self.get_target_commands(drone, target_m)
        
        # Delegate to the extracted controller to get the actual motor thrusts needed to achieve the desired angle and thrust.
        return self.controller.get_motor_thrusts(
            drone=drone,
            target_x=target_x,
            target_y=target_y
        )