import math
from typing import Any

from src.drone import Drone


class HardcodedBrain:
    """Wektorowy algorytm sterujący z kontrolerem PD (z płynnym hamowaniem)."""

    def activate(self, drone: Drone, target_m: tuple[float, float]) -> list[float]:
        dx = target_m[0] - drone._x
        dy = target_m[1] - drone._y

        # =========================================================
        # 1. FIZYKA: OBLICZENIE IDEALNEGO ZAWISU
        # =========================================================
        gravity_force = drone.mass * drone.gravity
        max_total_thrust = 2 * drone.max_thrust
        base_hover = gravity_force / max_total_thrust
        dist_to_target = math.hypot(dx, dy)

        # =========================================================
        # 2. OMIJANIE PRZESZKÓD (Pola Odpychające z radarem prędkości)
        # =========================================================
        repulsive_x = 0.0
        repulsive_y = 0.0

        drone_radius = drone.width_m / 2.0
        base_safe_dist = drone_radius * 1.5
        hard_limit = drone_radius * 1.1

        # 2. FIZYCZNE PRZYSPIESZENIE HAMUJĄCE
        # Wiemy, że ucięliśmy max wychylenie (lateral_push) do 0.2 (czyli 20% mocy silników).
        # max_thrust obu silników to 2 * drone.max_thrust.
        # Zatem maksymalna siła boczna, jaką dron może zahamować to:
        max_emergency_tilt = 0.6
        max_lateral_force = (2 * drone.max_thrust) * 0.2

        # Z 2 zasady dynamiki Newtona (a = F/m) wyliczamy maksymalne opóźnienie [m/s^2]:
        max_lateral_accel = max_lateral_force / drone.mass

        avoid_gain = 1.0  # Siła odpychania od samej bliskości
        brake_gain = 0.4  # NOWE: Siła hamowania awaryjnego (reakcja na prędkość)

        # Sprawdzamy, czy dron zainicjował już swoje sensory
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

                # 4. MODYFIKACJA GNRON (Teraz uwzględnia prędkość!)
                parking_zone_m = 1.0
                dist_factor = min(1.0, dist_to_target / parking_zone_m)

                # Obliczamy całkowitą prędkość drona
                current_speed = math.hypot(drone._vel_x, drone._vel_y)
                # Jeśli leci szybciej niż 1.5 m/s, speed_factor dąży do 1.0 (zabraniamy ignorować ściany)
                speed_factor = min(1.0, current_speed / 1.5)

                # Aby zignorować ścianę, dron musi być OBA: blisko celu ORAZ lecieć wolno.
                # Bierzemy max(), co oznacza, że duża prędkość "nadpisze" małą odległość do celu.
                goal_factor = max(dist_factor, speed_factor)

                # 5. TWARDY LIMIT FIZYCZNY (Panic Override)
                if dist < base_safe_dist:
                    panic_factor = 1.0 - (dist - hard_limit) / (
                        base_safe_dist - hard_limit
                    )
                    panic_factor = max(0.0, min(1.0, panic_factor))
                else:
                    panic_factor = 0.0

                final_factor = max(goal_factor, panic_factor)
                # Aplikujemy ostateczny mnożnik!
                strength = raw_strength * final_factor

                repulsive_x -= dir_x * strength
                repulsive_y += dir_y * strength
        # =========================================================
        # 3. KONTROLER PD + ODPYCHANIE
        # =========================================================
        p_gain = 0.16
        d_gain = 0.08

        # Najpierw liczymy sam ciąg do celu z hamowaniem
        base_pull_x = (dx * p_gain) - (drone._vel_x * d_gain)

        # OGRANICZAMY chęć lotu do celu (żeby odpychanie ścian łatwo to pokonało)
        # Ograniczamy sam "gaz" do max 0.15 z mocy drona.
        base_pull_x = max(-0.15, min(0.15, base_pull_x))

        # TERAZ dodajemy odpychanie od przeszkód. Jeśli odpychanie jest duże (np. 0.5),
        # bez problemu nadpisze nasze nędzne 0.15 ciągu do celu!
        lateral_push = base_pull_x + repulsive_x

        # Ostateczny, fizyczny limit przechyłu drona poszerzamy ciut bardziej,
        # żeby omijając kolizję mógł rzucić się mocniej w bok
        lateral_push = max(-max_emergency_tilt, min(max_emergency_tilt, lateral_push))

        # To samo robimy dla osi pionowej:
        base_pull_y = base_hover - (dy * p_gain) + (drone._vel_y * d_gain)
        base_pull_y = max(base_hover * 0.5, min(1.0, base_pull_y))

        upward_push = base_pull_y + repulsive_y
        upward_push = max(base_hover * 0.2, min(1.0, upward_push))
        # =========================================================
        # 4. KINEMATYKA: POŻĄDANY KĄT I CIĄG
        # =========================================================
        target_angle_rad = math.atan2(lateral_push, upward_push)
        target_angle_deg = math.degrees(target_angle_rad)

        current_angle_deg = math.degrees(drone._angle) % 360
        if current_angle_deg > 180:
            current_angle_deg -= 360

        angle_diff = (target_angle_deg - current_angle_deg + 180) % 360 - 180
        thrust_magnitude = math.hypot(lateral_push, upward_push)

        # =========================================================
        # 5. ROZDZIAŁ MOCY NA SILNIKI - STROJENIE KĄTA
        # =========================================================
        # Zmniejszamy drastycznie czułość skrętu (P), żeby silniki nadążały!
        # Podbijamy hamowanie (D), żeby gładko parkował na zadanym kącie.
        turn_p_gain = 0.005  # Było 0.015 (3x słabiej)
        turn_d_gain = 0.010  # Było 0.005 (2x mocniej)

        turn_force = (angle_diff * turn_p_gain) - (drone._angular_vel * turn_d_gain)
        turn_force = max(-0.2, min(0.2, turn_force))  # Mniejszy max ciąg kierunkowy

        l_thrust = thrust_magnitude + turn_force
        r_thrust = thrust_magnitude - turn_force
        return [l_thrust, r_thrust]

