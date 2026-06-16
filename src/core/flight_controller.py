import math

from src.core.drone import Drone

class FlightController:
    """Low level PID controller converting desired flight vector to motor thrusts."""
    
    def __init__(self, turn_p_gain: float = 0.015, turn_d_gain: float = 0.010, max_turn_force: float = 0.6) -> None:
        # Store PID parameters as class properties
        self.turn_p_gain = turn_p_gain
        self.turn_d_gain = turn_d_gain
        self.max_turn_force = max_turn_force

    def get_motor_thrusts(
        self, 
        drone: Drone,
        target_x: float, 
        target_y: float,
        max_tilt_deg: float = 45.0
    ) -> list[float]:
        """
        Changes the direction/power vector (x, y) into direct left and right motor thrusts.
        Input range: target_x [-1.0, 1.0], target_y [-1.0, 1.0].
        """

        # Limit input
        target_x = max(-1.0, min(1.0, target_x))
        target_y = max(-1.0, min(1.0, target_y))
        
        # Calculate ideal hover for this specific drone
        base_hover = (drone.mass * drone.gravity) / (2 * drone.max_thrust)

        # 1. TARGET KINEMATICS
        # X axis directly maps to angle (e.g., -45 to 45 degrees)
        target_angle_rad = target_x * math.radians(max_tilt_deg)

        # Y axis controls thrust relative to hover
        if target_y < 0:
            # Ascend (negative Y): from hover up to 100% power
            target_thrust = base_hover + abs(target_y) * (1.0 - base_hover)
        else:
            # Descend (positive Y): from hover down to 0% power
            target_thrust = base_hover - target_y * base_hover

        target_thrust = max(0.0, min(1.0, target_thrust))

        # 2. TURN PID CONTROLLER
        target_angle_deg = math.degrees(target_angle_rad)
        current_angle_deg = math.degrees(drone._angle) % 360
        if current_angle_deg > 180:
            current_angle_deg -= 360

        angle_diff = (target_angle_deg - current_angle_deg + 180) % 360 - 180

        # Use the stored parameters and the angular velocity directly from the drone
        turn_force = (angle_diff * self.turn_p_gain) - (drone._angular_vel * self.turn_d_gain)
        turn_force = max(-self.max_turn_force, min(self.max_turn_force, turn_force))

        # 3. MIXING THRUST
        l_thrust = target_thrust + turn_force
        r_thrust = target_thrust - turn_force

        return [l_thrust, r_thrust]