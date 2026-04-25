import math
import pygame
import numpy as np

from .utils import draw_vector
from .constants import *


class Drone:
    def __init__(self, x: float, y: float) -> None:
        self.x: float = x
        self.y: float = y
        self.vel_x: float = 0.0
        self.vel_y: float = 0.0
        self.angle: float = 0.0
        self.width: float = 50.0
        self.height: float = 15.0
        self.l_thrust: float = 0.0
        self.r_thrust: float = 0.0
        self.prev_l_thrust: float = 0.0
        self.prev_r_thrust: float = 0.0
        self.angular_vel: float = 0.0

        self.hover_frames: int = 0
        self.idle_frames: int = 0
        self.initial_dist: float = 0
        self.min_dist: float = 0
        self.has_touched_target: bool = False
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
        self.radar_range: float = 150

        self.last_sensor_data: list[float]

    def get_sensor_data(
        self, screen_width: int, screen_height: int, obstacles
    ) -> list[float]:
        """
        Wystrzeliwuje promienie i sprawdza odległość do krawędzi ekranu.
        Zwraca listę odległości.
        """
        distances: list[float] = []

        for s_angle in self.sensor_angles:
            # Obliczamy kąt promienia w świecie (kąt drona + offset sensora)
            # -90 bo w pygame 0 stopni to 'w prawo', a my chcemy 'w górę'
            rad: float = math.radians(self.angle + s_angle - 90)
            # moved cos and sin outside of the loop
            cos_rad = math.cos(rad)
            sin_rad = math.sin(rad)

            found = False
            # Raycasting: sprawdzamy punkty co 5 pikseli wzdłuż linii
            # removed np.arange to try and increase performance
            d = 0.0
            while d < self.max_sensor_dist:
                check_x: float = self.x + cos_rad * d
                check_y: float = self.y + sin_rad * d

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

                # Sprawdzanie przeszkód
                # collidepoint jest bardzo szybką funkcją w Pygame
                hit_obstacle = False
                for obs in obstacles:
                    if obs.collidepoint(check_x, check_y):
                        hit_obstacle = True
                        break

                if hit_obstacle:
                    distances.append(d)
                    found = True
                    break

                d += 5.0

            if not found:
                distances.append(self.max_sensor_dist)

        return distances

    def apply_thrust(self, left_power: float, right_power: float) -> None:
        rad: float = math.radians(self.angle - 90)
        total_thrust: float = (left_power + right_power) * THRUST_POWER
        self.vel_x += total_thrust * math.cos(rad)
        self.vel_y += total_thrust * math.sin(rad)
        self.angle += (left_power - right_power) * 4.0

        self.angle %= 360.0

    def update(self) -> None:
        self.vel_y += GRAVITY
        self.x += self.vel_x
        self.y += self.vel_y
        self.vel_x *= DRAG
        self.vel_y *= DRAG
        self.prev_l_thrust = self.l_thrust
        self.prev_r_thrust = self.r_thrust
        # FIZYKA OBROTU:
        # Różnica ciągu silników tworzy moment obrotowy
        torque = (
            self.r_thrust - self.l_thrust
        ) * 0.5  # 0.5 to siła obrotu, dostosuj ją
        # Prędkość kątowa rośnie pod wpływem momentu
        self.angular_vel += torque
        # Tłumienie obrotu (opór powietrza), żeby dron nie kręcił się w nieskończoność
        self.angular_vel *= 0.9
        # Aktualizacja kąta drona
        self.angle += self.angular_vel

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

        # --- 0. RYSOWANIE RADARU PRECYZJI (Tło pod dronem) ---
        # Obliczamy odległość do celu i aktywację radaru (dokładnie tak jak w get_inputs)
        dist_to_target = math.hypot(target_pos[0] - self.x, target_pos[1] - self.y)
        radar_val = max(0.0, 1.0 - (dist_to_target / self.radar_range))

        if radar_val > 0:
            # Tworzymy osobną powierzchnię z kanałem Alpha (przezroczystością)
            # Rozmiar to 2x promień radaru
            r_size = int(self.radar_range * 2)
            radar_surf = pygame.Surface((r_size, r_size), pygame.SRCALPHA)

            # Alfa (przezroczystość) zależy od aktywacji radaru.
            # Mnożymy przez 150, żeby przy maksymalnej bliskości nie zasłoniło całkiem tła
            alpha_fill = int(150 * radar_val)

            # Rysujemy półprzezroczyste wypełnienie
            _ = pygame.draw.circle(
                radar_surf,
                (0, 255, 100, alpha_fill),
                (int(self.radar_range), int(self.radar_range)),
                int(self.radar_range),
            )

            # Rysujemy wyraźną obwódkę radaru (zmienia grubość gdy dron jest blisko!)
            border_thickness = max(1, int(3 * radar_val))
            _ = pygame.draw.circle(
                radar_surf,
                (0, 255, 100, 255),
                (int(self.radar_range), int(self.radar_range)),
                int(self.radar_range),
                border_thickness,
            )

            # Nakładamy powierzchnię radaru na ekran (centrujemy na dronie)
            _ = screen.blit(
                radar_surf,
                (int(self.x - self.radar_range), int(self.y - self.radar_range)),
            )

        # --- 1. RYSOWANIE SENSORÓW (Warstwa najniższa) ---
        if hasattr(self, "last_sensor_data"):
            for i, dist in enumerate(self.last_sensor_data):
                # Kąt sensora w świecie
                rad = math.radians(self.angle + self.sensor_angles[i] - 90)
                end_x = self.x + math.cos(rad) * dist
                end_y = self.y + math.sin(rad) * dist

                # Kolor zależy od odległości (czerwieńszy im bliżej ściany)
                color_val = max(
                    0, min(255, int(255 * (1 - dist / self.max_sensor_dist)))
                )
                s_color = (
                    100 + color_val // 2,
                    100 - color_val // 3,
                    100 - color_val // 3,
                )

                _ = pygame.draw.line(
                    screen, s_color, (self.x, self.y), (end_x, end_y), 1
                )
                if dist < self.max_sensor_dist:
                    _ = pygame.draw.circle(
                        screen, (255, 0, 0), (int(end_x), int(end_y)), 3
                    )

        # --- 2. RYSOWANIE WEKTORA DO CELU ---
        #
        # UWAGA: dron nie powinien rysować celu, wektora do celu też raczej nie
        # Cienka przerywana linia lub kropka wskazująca cel
        # _ = pygame.draw.circle(screen, (0, 255, 0), target_pos, 8, 2)  # Cel jako kółko
        # _ = pygame.draw.line(
        #    screen, (0, 100, 0), (self.x, self.y), target_pos, 1
        # )  # Linia pomocnicza

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
        self,
        target_x: float,
        target_y: float,
        screen_width: int,
        screen_height: int,
        obstacles,
    ) -> list[float]:
        # 1. Odległości z 8 sensorów (znormalizowane 0.0 - 1.0)
        sensors: list[float] = self.get_sensor_data(
            screen_width, screen_height, obstacles
        )

        self.last_sensor_data = sensors

        normalized_sensors: list[float] = [d / self.max_sensor_dist for d in sensors]

        # 2. Prędkość relatywna (Body-Fixed Frame)
        speed = math.hypot(self.vel_x, self.vel_y)
        # Normalizujemy prędkość (np. do zakresu 0 - 1, zakładając max 15)
        norm_speed = min(1.0, speed / 15.0)

        # Kąt wektora prędkości w świecie
        vel_angle_rad = math.atan2(self.vel_y, self.vel_x)
        # Kąt drona w świecie (pamiętamy o przesunięciu wizualnym -90 jeśli go używamy)
        drone_angle_rad = math.radians(self.angle - 90)

        # Różnica kątów (Gdzie lecę względem tego, gdzie patrzę)
        rel_vel_angle = vel_angle_rad - drone_angle_rad

        # Zwijamy do [-pi, pi]
        rel_vel_angle = (rel_vel_angle + math.pi) % (2 * math.pi) - math.pi

        # Używamy Sin i Cos, żeby uniknąć skoków wartości
        v_sin = math.sin(rel_vel_angle)
        v_cos = math.cos(rel_vel_angle)

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

        # radar
        precision_radar = max(0.0, 1.0 - (dist / self.radar_range))

        # B) Kąt do celu względem tego, gdzie "patrzy" dron
        target_angle_rad = math.atan2(dy, dx)

        # drone angle rad obliczony wcześniej
        # Obliczamy różnicę kątów
        diff_angle = target_angle_rad - drone_angle_rad

        # "Zwijamy" kąt do przedziału [-π, π], żeby dron wiedział, czy cel
        # jest szybciej osiągalny kręcąc się w lewo, czy w prawo.
        diff_angle = (diff_angle + math.pi) % (2 * math.pi) - math.pi

        sin_targ = math.sin(diff_angle)
        cos_targ = math.cos(diff_angle)

        norm_angular_vel = math.tanh(self.angular_vel / 5.0)

        # Łączymy wszystko w jedną listę (Input Layer)
        # Dzięki temu nadal przekazujemy dokładnie 14 wartości!
        return normalized_sensors + [
            norm_speed,
            v_sin,
            v_cos,
            s_angle,
            c_angle,
            norm_dist,
            precision_radar,
            sin_targ,
            cos_targ,
            self.l_thrust,
            self.r_thrust,
            norm_angular_vel,
        ]
