import math
import pygame
import numpy as np

from .utils import draw_vector
from .constants import *


class Drone:
    def __init__(self, x: float, y: float) -> None:
        self.x: float = x
        self.y: float = y
        self.vel_x: float = 0
        self.vel_y: float = 0
        self.angle: float = 0
        self.width: float = 50
        self.height: float = 15

        self.hover_frames: int = 0
        # Sensory: określamy kąty, pod jakimi dron "patrzy"
        self.sensor_angles: list[float] = [
            0.0,
            45.0,
            90.0,
            135.0,
            180.0,
            225.0,
            270.0,
            315.0,
        ]
        self.max_sensor_dist: float = 250  # Jak daleko dron widzi

    def get_sensor_data(self, screen_width: int, screen_height: int) -> list[float]:
        """
        Wystrzeliwuje promienie i sprawdza odległość do krawędzi ekranu.
        Zwraca listę odległości.
        """
        distances: list[float] = []

        for s_angle in self.sensor_angles:
            # Obliczamy kąt promienia w świecie (kąt drona + offset sensora)
            # -90 bo w pygame 0 stopni to 'w prawo', a my chcemy 'w górę'
            rad: float = math.radians(self.angle + s_angle - 90)

            found = False
            # Raycasting: sprawdzamy punkty co 5 pikseli wzdłuż linii
            for d in np.arange(0.0, self.max_sensor_dist, 5):
                check_x: float = self.x + math.cos(rad) * d
                check_y: float = self.y + math.sin(rad) * d

                # Sprawdzamy kolizję z krawędziami okna
                if (
                    check_x <= 0
                    or check_x >= screen_width
                    or check_y <= 0
                    or check_y >= screen_height
                ):
                    distances.append(d)
                    found = True
                    break

            if not found:
                distances.append(self.max_sensor_dist)

        return distances

    def apply_thrust(self, left_power: float, right_power: float) -> None:
        rad: float = math.radians(self.angle - 90)
        total_thrust: float = (left_power + right_power) * THRUST_POWER
        self.vel_x += total_thrust * math.cos(rad)
        self.vel_y += total_thrust * math.sin(rad)
        self.angle += (left_power - right_power) * 4.0

    def update(self) -> None:
        self.vel_y += GRAVITY
        self.x += self.vel_x
        self.y += self.vel_y
        self.vel_x *= DRAG
        self.vel_y *= DRAG

    def draw(
        self,
        screen: pygame.Surface,
        target_pos: tuple[int, int],
        left_p: float,
        right_p: float,
    ):
        """
        Pełna wizualizacja drona, jego zmysłów i działań.
        left_p i right_p to aktualna moc silników (0.0 do 1.0).
        """

        # --- 1. RYSOWANIE SENSORÓW (Warstwa najniższa) ---
        sensors = self.get_sensor_data(SCREEN_WIDTH, SCREEN_HEIGHT)
        for i, dist in enumerate(sensors):
            # Kąt sensora w świecie
            rad = math.radians(self.angle + self.sensor_angles[i] - 90)
            end_x = self.x + math.cos(rad) * dist
            end_y = self.y + math.sin(rad) * dist

            # Kolor zależy od odległości (czerwieńszy im bliżej ściany)
            color_val = max(0, min(255, int(255 * (1 - dist / self.max_sensor_dist))))
            s_color = (100 + color_val // 2, 100 - color_val // 3, 100 - color_val // 3)

            _ = pygame.draw.line(screen, s_color, (self.x, self.y), (end_x, end_y), 1)
            if dist < self.max_sensor_dist:
                _ = pygame.draw.circle(screen, (255, 0, 0), (int(end_x), int(end_y)), 3)

        # --- 2. RYSOWANIE WEKTORA DO CELU ---
        # Cienka przerywana linia lub kropka wskazująca cel
        _ = pygame.draw.circle(screen, (0, 255, 0), target_pos, 8, 2)  # Cel jako kółko
        _ = pygame.draw.line(
            screen, (0, 100, 0), (self.x, self.y), target_pos, 1
        )  # Linia pomocnicza

        # --- 3. RYSOWANIE WEKTORÓW SIŁY SILNIKÓW ---
        # Obliczamy pozycje silników względem środka drona (obrócone)
        rad_drone = math.radians(self.angle)
        # Silniki są przesunięte o 25px w lewo i prawo od środka drona
        l_eng_x = self.x + math.cos(rad_drone + math.pi) * 25
        l_eng_y = self.y + math.sin(rad_drone + math.pi) * 25
        r_eng_x = self.x + math.cos(rad_drone) * 25
        r_eng_y = self.y + math.sin(rad_drone) * 25

        # Wektor siły: rysujemy go w dół (jako odrzut), co wizualnie pcha drona w górę
        # Długość wektora zależy od mocy (max 50px)
        vector_scale = 50
        draw_vector(
            screen, (l_eng_x, l_eng_y), self.angle, left_p * vector_scale, (255, 165, 0)
        )
        draw_vector(
            screen,
            (r_eng_x, r_eng_y),
            self.angle,
            right_p * vector_scale,
            (255, 165, 0),
        )

        # --- 4. RYSOWANIE KADŁUBA DRONA (Warstwa najwyższa) ---
        # Tworzymy powierzchnię drona (szerokość 60, wysokość 15)
        drone_surf = pygame.Surface((60, 20), pygame.SRCALPHA)

        # Centralny kadłub
        _ = pygame.draw.rect(drone_surf, (60, 60, 70), (10, 5, 40, 10), border_radius=4)
        # Silniki (wizualne bloki)
        _ = pygame.draw.rect(
            drone_surf, (0, 200, 255), (0, 0, 10, 20), border_radius=2
        )  # Lewy
        _ = pygame.draw.rect(
            drone_surf, (0, 200, 255), (50, 0, 10, 20), border_radius=2
        )  # Prawy
        # "Oczko" z przodu, żeby było widać gdzie jest przód drona
        _ = pygame.draw.circle(drone_surf, (255, 255, 255), (30, 5), 2)

        # Obracamy drona
        rotated_drone = pygame.transform.rotate(drone_surf, -self.angle)
        rect = rotated_drone.get_rect(center=(int(self.x), int(self.y)))
        _ = screen.blit(rotated_drone, rect.topleft)

    def get_inputs(
        self, target_x: float, target_y: float, screen_width: int, screen_height: int
    ) -> list[float]:
        # 1. Odległości z 8 sensorów (znormalizowane 0.0 - 1.0)
        sensors: list[float] = self.get_sensor_data(screen_width, screen_height)
        normalized_sensors: list[float] = [d / self.max_sensor_dist for d in sensors]

        # 2. Prędkość (zakładamy max prędkość ok. 15, skalujemy do [-1, 1])
        # Używamy math.tanh lub zwykłego dzielenia z clipem
        nv_x = max(-1.0, min(1.0, self.vel_x / 10.0))
        nv_y = max(-1.0, min(1.0, self.vel_y / 10.0))

        # 3. Orientacja (Sin i Cos kąta - zapobiega przeskokowi 359->0)
        rad = math.radians(self.angle)
        s_angle = math.sin(rad)
        c_angle = math.cos(rad)

        # 4. Odległość i kąt do celu (zamiast surowego dx, dy)
        dx = target_x - self.x
        dy = target_y - self.y

        # A) Znormalizowana odległość (0.0 to środek celu, 1.0 to przekątna ekranu)
        max_dist = math.hypot(screen_width, screen_height)
        dist = math.hypot(dx, dy)
        norm_dist = dist / max_dist

        # B) Kąt do celu względem tego, gdzie "patrzy" dron
        target_angle_rad = math.atan2(dy, dx)

        # Pamiętamy o przesunięciu -90 stopni, które masz w logice lotu drona
        drone_angle_rad = math.radians(self.angle - 90)

        # Obliczamy różnicę kątów
        diff_angle = target_angle_rad - drone_angle_rad

        # "Zwijamy" kąt do przedziału [-π, π], żeby dron wiedział, czy cel
        # jest szybciej osiągalny kręcąc się w lewo, czy w prawo.
        diff_angle = (diff_angle + math.pi) % (2 * math.pi) - math.pi

        # Skalujemy do [-1.0, 1.0] (ideał dla funkcji aktywacji sigmoid)
        norm_angle = diff_angle / math.pi

        # Łączymy wszystko w jedną listę (Input Layer)
        # Dzięki temu nadal przekazujemy dokładnie 14 wartości!
        return normalized_sensors + [
            nv_x,
            nv_y,
            s_angle,
            c_angle,
            norm_dist,
            norm_angle,
        ]
