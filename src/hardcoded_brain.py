# src/hardcoded_brain.py
import math
from typing import Any


class HardcodedBrain:
    """Naiwny algorytm sterujący dronem (Baseline / Heurystyka)."""

    def activate(self, drone: Any, target_m: tuple[float, float]) -> list[float]:
        # 1. Wektor do celu (w metrach)
        dx = target_m[0] - drone._x
        dy = target_m[1] - drone._y
        dist = math.hypot(dx, dy)

        # 2. Obliczamy kąt do celu (w radianach)
        # UWAGA: W Pygame Y rośnie w dół, więc dla math.atan2 często odwraca się znak Y
        target_angle_rad = math.atan2(-dy, dx)
        target_angle_deg = math.degrees(target_angle_rad)

        # 3. Kąt drona (normalizujemy do przedziału -180 do 180)
        current_angle = drone._angle % 360
        if current_angle > 180:
            current_angle -= 360

        # 4. Różnica kątów (najkrótsza droga obrotu)
        angle_diff = (target_angle_deg - current_angle + 180) % 360 - 180

        # --- LOGIKA STEROWANIA ---

        # Ciąg bazowy (zakładamy bazową chęć lotu w górę/przód)
        base_thrust = 0.1

        # Zwalniamy przed celem (hamowanie)
        if dist < 2.0:  # Mniej niż 2 metry
            base_thrust = 0.18  # Moc pozwalająca na delikatne opadanie/zawis

        # Dodatkowa logika: Jeśli cel jest pod nami (dy > 0), wyłącz silniki, niech spada!
        if dy > 0 and dist > 1.0:
            base_thrust = 0.0

        # Sterowanie obrotem (Regulator Proporcjonalny)
        # Mnożnik 0.02 to siła skrętu. Im większa, tym szybciej reaguje.
        turn_force = angle_diff * 0.02

        # Zabezpieczenie, żeby ciąg różnicowy nie był zbyt agresywny
        turn_force = max(-0.4, min(0.4, turn_force))

        # Dodajemy moc na jeden silnik, ujmujemy z drugiego
        # (Zmień + i - miejscami, jeśli dron zacznie odwracać się tyłem do celu!)
        l_thrust = base_thrust - turn_force
        r_thrust = base_thrust + turn_force

        # Na koniec "obcinamy" wartości do przedziału [0.0, 1.0] jak w sieci NEAT
        l_thrust = max(0.0, min(1.0, l_thrust))
        r_thrust = max(0.0, min(1.0, r_thrust))

        return [l_thrust, r_thrust]
