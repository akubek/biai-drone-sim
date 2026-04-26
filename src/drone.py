import math
import pygame

from .utils import draw_vector
from .constants import *


class Drone:
    def __init__(
        self,
        # starting position
        start_x_m: float,
        start_y_m: float,
        # physical drone parameters
        mass: float = 0.8,  # [kg]
        width_m: float = 0.35,  # [m] - 35 cm szerokości
        height_m: float = 0.1,  # [m] - 10 cm wysokości
        max_thrust: float = 15.0,  # [N] - max ciąg na JEDEN silnik (razem 30N)        torque_power: float = 0.5,
        engine_offset_m: float = 0.175,  # [m] - ramię siły (odległość silnika od środka). Domyślnie połowa szerokości.
        engine_response_rate: float = 60.0,  # response rate const Hz
        # physics simulation parameters
        gravity: float = 9.81,  # [m/s^2]
        drag_coeff: float = 0.5,  # [kg/s]        angular_drag: float = 0.0,
        angular_drag: float = 0.05,  # [kg*m^2/s] - opór powietrza przy obracaniu się (żeby dron nie kręcił się w nieskończoność)
        # sensor parameters
        distance_sensor_count: int = 8,  # amount of distance sensord (equally distributed)
        radar_range: float = 1.0,  # [m]
        close_radar: float = 0.35,
        max_sensor_dist: float = 1.25,  # [m]
        raycast_step_m: float = 0.05,  # [m]
        PPM: float = 200,  # pixels per meter
    ) -> None:
        self.mass: float = mass
        self.width_m: float = width_m
        self.height_m: float = height_m
        self.engine_offset_m: float = engine_offset_m
        self.engine_response_rate: float = engine_response_rate
        self.max_thrust: float = max_thrust
        self.gravity: float = gravity
        self.drag_coeff: float = drag_coeff
        self.angular_drag: float = angular_drag

        self.radar_range: float = radar_range
        self.close_radar: float = close_radar
        self.max_sensor_dist: float = max_sensor_dist
        self.raycast_step_m: float = raycast_step_m

        self.distance_sensor_count: int = distance_sensor_count

        # sensor angles in radians
        self.sensor_angles: list[float] = [
            i * (2 * math.pi / self.distance_sensor_count)
            for i in range(self.distance_sensor_count)
        ]

        self.last_sensor_data: list[float]

        # Moment bezwładności dla pręta: I = (1/12) * m * L^2
        # To fizycznie opisuje, jak trudno obrócić drona.
        self.inertia: float = (1.0 / 12.0) * self.mass * (self.width_m**2)

        # --- ZMIENNE STANU (Zaczynamy od '_' by zaznaczyć, że to prywatne parametry fizyczne) ---
        self._x: float = start_x_m  # [m]
        self._y: float = start_y_m  # [m]
        self._vel_x: float = 0.0  # [m/s]
        self._vel_y: float = 0.0  # [m/s]

        self._angle: float = 0.0  # [rad] - UWAGA: teraz obrót trzymamy w radianach!
        self._angular_vel: float = 0.0  # [rad/s]

        # Bieżący ciąg zadany przez AI (0.0 do 1.0)
        self.actual_l_thrust: float = 0.0
        self.actual_r_thrust: float = 0.0

        self.l_command: float = 0.0
        self.r_command: float = 0.0
        self.prev_l_command: float = 0.0
        self.prev_r_command: float = 0.0

        # --- PRE-RENDER DRONE ---
        px_width: int = int(self.width_m * PPM)  # for default values will be 70 px
        px_height: int = int(self.height_m * PPM)  # and 20 px
        self._base_surf: pygame.Surface = pygame.Surface(
            (px_width, px_height), pygame.SRCALPHA
        )
        # Centralny kadłub
        _ = pygame.draw.rect(
            self._base_surf,
            (60, 60, 70),
            (px_width * 0.2, px_height * 0.25, px_width * 0.6, px_height * 0.5),
            border_radius=4,
        )
        # Silniki
        _ = pygame.draw.rect(
            self._base_surf,
            (0, 200, 255),
            (0, 0, px_width * 0.15, px_height),
            border_radius=2,
        )
        _ = pygame.draw.rect(
            self._base_surf,
            (0, 200, 255),
            (px_width * 0.85, 0, px_width * 0.15, px_height),
            border_radius=2,
        )
        # "Oczko" z przodu (środek górnej krawędzi)
        _ = pygame.draw.circle(
            self._base_surf, (255, 255, 255), (px_width / 2, px_height * 0.25), 2
        )

        # --- PRE-RENDER RADAR SURFACE ---
        # Tworzymy płótno dla radaru tylko raz, by uniknąć alokacji pamięci co klatkę
        r_size_px = int(self.radar_range * PPM * 2)
        self._radar_surf: pygame.Surface = pygame.Surface(
            (r_size_px, r_size_px), pygame.SRCALPHA
        )
        self._radar_radius_px: int = int(self.radar_range * PPM)

    def get_sensor_data(
        self,
        screen_width: int,
        screen_height: int,
        obstacles: list[pygame.Rect],
        PPM: float,  # pixels per meter in the simulation
    ) -> list[float]:
        """
        Zoptymalizowany Raycasting używający Pygame C-API.
        Zwraca listę odległości w METRACH.
        """
        distances: list[float] = []

        start_x_px: float = self._x * PPM
        start_y_px: float = self._y * PPM
        max_dist_px: float = self.max_sensor_dist * PPM

        # Reprezentacja krawędzi ekranu jako prostokąt
        screen_rect: pygame.Rect = pygame.Rect(0, 0, screen_width, screen_height)

        for s_angle in self.sensor_angles:
            # sensor ray direction
            rad: float = self._angle + s_angle  # moved cos and sin outside of the loop

            # direction vector
            dir_x: float = math.sin(rad)
            dir_y: float = -math.cos(rad)

            # sesor ray end point
            end_x_px: float = start_x_px + dir_x * max_dist_px
            end_y_px: float = start_y_px + dir_y * max_dist_px

            # starting shortest distance = max distance
            closest_dist_px: float = max_dist_px
            # --- A) KOLIZJA Z EKRANEM ---
            # clipline zwraca odcinek, który mieści się W prostokącie.
            # Jeśli promień wychodzi za ekran, clipped_screen[1] to punkt uderzenia w ścianę!
            clipped_screen = screen_rect.clipline(
                start_x_px, start_y_px, end_x_px, end_y_px
            )
            if clipped_screen:
                exit_point = clipped_screen[1]
                dist_to_edge = math.hypot(
                    exit_point[0] - start_x_px, exit_point[1] - start_y_px
                )
                if dist_to_edge < closest_dist_px:
                    closest_dist_px = dist_to_edge
            else:
                # Dron znajduje się całkowicie poza ekranem w miejscu startu (kara w ewolucji go zdejmie)
                closest_dist_px: float = 0.0

            # --- B) KOLIZJA Z PRZESZKODAMI ---
            for obs in obstacles:
                # Sprawdzamy geometryczne przecięcie linii promienia z Rectem przeszkody
                clipped = obs.clipline(start_x_px, start_y_px, end_x_px, end_y_px)
                if clipped:
                    # clipped[0] to punkt, w którym promień wchodzi w przeszkodę
                    hit_x, hit_y = clipped[0]
                    dist_to_obs = math.hypot(hit_x - start_x_px, hit_y - start_y_px)
                    if dist_to_obs < closest_dist_px:
                        closest_dist_px = dist_to_obs

            # 3. Zapisz i przelicz wynik Z POWROTEM NA METRY!
            final_dist_m: float = closest_dist_px / PPM
            distances.append(final_dist_m)

        self.last_sensor_data = distances

        return distances

    def set_engine_thrust(self, left_cmd: float, right_cmd: float) -> None:
        """
        Zapisuje stan silników na podstawie wyjścia sieci NEAT.
        Wartości wejściowe (0.0 do 1.0) są bezpiecznie przycinane (clamping).
        """
        self.prev_l_command = self.l_command
        self.prev_r_command = self.r_command

        self.l_command = max(0.0, min(1.0, left_cmd))
        self.r_command = max(0.0, min(1.0, right_cmd))

    def update(self, dt: float = 1.0 / 60.0) -> None:
        """
        Główna pętla fizyki drona. Aktualizuje prędkość i pozycję na podstawie sił (SI).
        dt to czas trwania klatki (Delta Time).
        """

        alpha = 1.0 - math.exp(-self.engine_response_rate * dt)

        self.actual_l_thrust += (self.l_command - self.actual_l_thrust) * alpha
        self.actual_r_thrust += (self.r_command - self.actual_r_thrust) * alpha

        # self.actual_l_thrust = self.l_command
        # self.actual_r_thrust = self.r_command

        # --- 1. SIŁA CIĄGU SILNIKÓW (Newnton) ---
        f_left = self.actual_l_thrust * self.max_thrust
        f_right = self.actual_r_thrust * self.max_thrust
        total_thrust = f_left + f_right

        # Wektory siły ciągu. 0 radianów = w górę ekranu (Y maleje, X stoi)
        thrust_vec_x = total_thrust * math.sin(self._angle)
        thrust_vec_y = -total_thrust * math.cos(self._angle)

        # --- 2. OPÓR POWIETRZA I GRAWITACJA (Newnton) ---
        drag_x = -self.drag_coeff * self._vel_x
        drag_y = -self.drag_coeff * self._vel_y
        gravity_force_y = (
            self.mass * self.gravity
        )  # Grawitacja ciągnie w dół (dodatnie Y)

        # --- 3. SUMA SIŁ LINIOWYCH ---
        net_force_x = thrust_vec_x + drag_x
        net_force_y = thrust_vec_y + drag_y + gravity_force_y

        # --- 4. RUCH LINIOWY (Druga zasada dynamiki Newtona: a = F/m) ---
        accel_x = net_force_x / self.mass
        accel_y = net_force_y / self.mass

        # Aktualizacja prędkości (Przyspieszenie * czas)
        self._vel_x += accel_x * dt
        self._vel_y += accel_y * dt

        # Aktualizacja pozycji (Prędkość * czas)
        self._x += self._vel_x * dt
        self._y += self._vel_y * dt

        # --- 5. FIZYKA OBROTU (Moment obrotowy: Torque) ---
        # Jeśli lewy silnik (f_left) pcha mocniej, dron obróci się w prawo (wartość dodatnia)
        # Ramię siły to nasza odległość silnika od środka (engine_offset_m)
        torque = (f_left - f_right) * self.engine_offset_m

        # Opór aerodynamiczny obrotu, żeby dron naturalnie wyhamowywał kręcenie się
        angular_drag_torque = -self.angular_drag * self._angular_vel
        net_torque = torque + angular_drag_torque

        # Przyspieszenie kątowe (Epsilon = Torque / Moment bezwładności)
        angular_accel = net_torque / self.inertia

        # Aktualizacja prędkości kątowej i kąta
        self._angular_vel += angular_accel * dt
        self._angle += self._angular_vel * dt

        # Zwijanie kąta, by zawsze był w przedziale [0, 2*PI] (zapobiega to problemom trygonometrycznym)
        self._angle %= 2 * math.pi

    # ==========================================
    # LOGIKA KOLIZJI (HITBOX)
    # ==========================================
    def check_collision(
        self,
        screen_width_px: int,
        screen_height_px: int,
        obstacles: list[pygame.Rect],
        PPM: float,
    ) -> bool:
        """
        Zwraca True, jeśli dron wyleciał za ekran lub uderzył w przeszkodę.
        Używa okręgu opisanego na szerokości drona dla super-szybkiej detekcji.
        """
        # Obliczamy pozycję drona w pikselach
        px_x = int(self._x * PPM)
        px_y = int(self._y * PPM)

        # Promień drona w pikselach (połowa szerokości) - simplified
        radius_px = int((self.width_m / 2) * PPM)

        # 1. Kolizja z krawędziami ekranu
        if (
            px_x - radius_px < 0
            or px_x + radius_px > screen_width_px
            or px_y - radius_px < 0
            or px_y + radius_px > screen_height_px
        ):
            return True

        # 2. Kolizja z przeszkodami (szybki test zderzenia okręgu z prostokątem)
        # Pygame obsługuje to najlepiej poprzez sprawdzenie dystansu od środka drona do najbliższego punktu na prostokącie.
        for obs in obstacles:
            closest_x = max(obs.left, min(px_x, obs.right))
            closest_y = max(obs.top, min(px_y, obs.bottom))

            dist_x = px_x - closest_x
            dist_y = px_y - closest_y

            if (dist_x**2 + dist_y**2) < (radius_px**2):
                return True

        return False

    # ==========================================
    # WIZUALIZACJA
    # ==========================================
    def draw(
        self,
        screen: pygame.Surface,
        target_pos_m: tuple[float, float],
        PPM: float,
        # Flagi kontrolujące co rysujemy
        show_radar: bool = False,
        show_sensors: bool = True,
        show_thrust: bool = True,
        show_hitbox: bool = False,
    ) -> None:
        """
        Pełna wizualizacja drona, jego zmysłów i działań.
        left_p i right_p to aktualna moc silników (0.0 do 1.0).
        """

        """Główna metoda rysująca, delegująca zadania do sub-metod."""
        px_x = int(self._x * PPM)
        px_y = int(self._y * PPM)

        if show_radar:
            self._draw_radar(screen, target_pos_m, px_x, px_y)

        if show_sensors:
            self._draw_sensors(screen, px_x, px_y, PPM)

        if show_thrust:
            self._draw_thrust(screen, px_x, px_y, PPM)

        # Kadłub rysujemy zawsze
        self._draw_body(screen, px_x, px_y)

        if show_hitbox:
            radius_px = int((self.width_m / 2) * PPM)
            _ = pygame.draw.circle(screen, (255, 0, 255), (px_x, px_y), radius_px, 1)

    def _draw_radar(
        self,
        screen: pygame.Surface,
        target_pos_m: tuple[float, float],
        px_x: int,
        px_y: int,
    ) -> None:
        dist_to_target_m = math.hypot(
            target_pos_m[0] - self._x, target_pos_m[1] - self._y
        )
        radar_val = max(0.0, 1.0 - (dist_to_target_m / self.radar_range))

        if radar_val > 0:
            # 1. ZAMIAST tworzyć nową powierzchnię, CZYŚCIMY starą (wypełniamy przezroczystością)
            _ = self._radar_surf.fill((0, 0, 0, 0))

            # 2. Rysujemy na wyczyszczonym płótnie
            alpha_fill = int(150 * radar_val)
            _ = pygame.draw.circle(
                self._radar_surf,
                (0, 255, 100, alpha_fill),
                (self._radar_radius_px, self._radar_radius_px),
                self._radar_radius_px,
            )

            border_thickness = max(1, int(3 * radar_val))
            _ = pygame.draw.circle(
                self._radar_surf,
                (0, 255, 100, 255),
                (self._radar_radius_px, self._radar_radius_px),
                self._radar_radius_px,
                border_thickness,
            )

            # 3. Szybki, bezpieczny blit na ekran główny
            _ = screen.blit(
                self._radar_surf,
                (px_x - self._radar_radius_px, px_y - self._radar_radius_px),
            )

    def _draw_sensors(
        self, screen: pygame.Surface, px_x: int, px_y: int, PPM: float
    ) -> None:
        if not hasattr(self, "last_sensor_data"):
            return

        for i, dist_m in enumerate(self.last_sensor_data):
            rad = self._angle + self.sensor_angles[i]

            dist_px = int(dist_m * PPM)
            end_x = px_x + math.sin(rad) * dist_px
            end_y = px_y - math.cos(rad) * dist_px

            color_val = max(0, min(255, int(255 * (1 - dist_m / self.max_sensor_dist))))
            s_color = (100 + color_val // 2, 100 - color_val // 3, 100 - color_val // 3)

            _ = pygame.draw.line(screen, s_color, (px_x, px_y), (end_x, end_y), 1)

            if dist_m < self.max_sensor_dist:
                _ = pygame.draw.circle(screen, (255, 0, 0), (int(end_x), int(end_y)), 3)

    def _draw_thrust(
        self, screen: pygame.Surface, px_x: int, px_y: int, PPM: float
    ) -> None:
        # Przeliczamy kąt fizyczny (radiany) na kąt Pygame (stopnie, przeciwnie do wskazówek zegara)
        deg_angle = math.degrees(-self._angle)

        # Pozycja silników w pikselach względem środka
        offset_px = self.engine_offset_m * PPM

        # Rotacja dla pozycji silników (Kąt + 90 stopni do osi drona)
        l_eng_x = px_x + math.cos(self._angle + math.pi) * offset_px
        l_eng_y = px_y + math.sin(self._angle + math.pi) * offset_px

        r_eng_x = px_x + math.cos(self._angle) * offset_px
        r_eng_y = px_y + math.sin(self._angle) * offset_px

        # Skala wizualna płomienia
        vector_scale = 50

        # Jeśli używasz starej funkcji `draw_vector`, która przyjmuje kąt w stopniach, musisz go przekazać odpowiednio
        draw_vector(
            screen,
            (l_eng_x, l_eng_y),
            -deg_angle,
            self.actual_l_thrust * vector_scale,
            (255, 165, 0),
        )
        draw_vector(
            screen,
            (r_eng_x, r_eng_y),
            -deg_angle,
            self.actual_r_thrust * vector_scale,
            (255, 165, 0),
        )

    def _draw_body(self, screen: pygame.Surface, px_x: int, px_y: int) -> None:
        # Obracamy wcześniej zapisaną statyczną powierzchnię (Kąt z minusem, bo Pygame obraca CCW)
        deg_angle = math.degrees(-self._angle)
        rotated_drone = pygame.transform.rotate(self._base_surf, deg_angle)
        rect = rotated_drone.get_rect(center=(px_x, px_y))
        _ = screen.blit(rotated_drone, rect.topleft)

    def get_inputs(
        self,
        target_pos_m: tuple[float, float],
        screen_width_px: int,
        screen_height_px: int,
        obstacles: list[pygame.Rect],
        PPM: float,
    ) -> list[float]:
        # distance sensor data
        sensors: list[float] = self.get_sensor_data(
            screen_width_px, screen_height_px, obstacles, PPM
        )

        # get_sensor_data, already saves last sensor_data
        # self.last_sensor_data = sensors
        # normalize sensor data [0,max length] -> [0,1]
        normalized_sensors: list[float] = [d / self.max_sensor_dist for d in sensors]

        # relative speed [m/s] (Body-Fixed Frame)
        speed_m_s = math.hypot(self._vel_x, self._vel_y)
        # normalize speed expecting 15 m/s to be the reasonable maximum
        norm_speed = min(1.0, speed_m_s / 15.0)

        # velocity angle in world view
        vel_angle_rad = math.atan2(self._vel_y, self._vel_x)

        # difference in angles
        rel_vel_angle = vel_angle_rad - self._angle

        # to range [-pi, pi]
        rel_vel_angle = (rel_vel_angle + math.pi) % (2 * math.pi) - math.pi

        # angle sin and cos - avoid sudden jump in the angle
        v_sin = math.sin(rel_vel_angle)
        v_cos = math.cos(rel_vel_angle)

        # world sin and cos
        s_angle = math.sin(self._angle)
        c_angle = math.cos(self._angle)

        # angle and distance to target in meters
        target_x_m, target_y_m = target_pos_m
        dx = target_x_m - self._x
        dy = target_y_m - self._y

        # normalized distnce (0.0 is center of the target, 1.0 is screen diagonal)
        max_dist_m = math.hypot(screen_width_px / PPM, screen_height_px / PPM)
        dist_m = math.hypot(dx, dy)
        norm_dist = dist_m / max_dist_m

        # radar
        proximity_radar = max(0.0, 1.0 - (dist_m / self.radar_range))
        proximity_radar2 = max(0.0, 1.0 - (dist_m / self.close_radar))

        # relative angle to target
        target_angle_rad = math.atan2(dy, dx)
        diff_angle = target_angle_rad - self._angle
        diff_angle = (diff_angle + math.pi) % (2 * math.pi) - math.pi  # to [-pi, pi]

        sin_targ = math.sin(diff_angle)
        cos_targ = math.cos(diff_angle)

        norm_angular_vel = math.tanh(self._angular_vel / 5.0)

        # Łączymy wszystko w jedną listę (Input Layer)
        return normalized_sensors + [
            norm_speed,
            v_sin,
            v_cos,
            s_angle,
            c_angle,
            norm_dist,
            proximity_radar,
            proximity_radar2,
            sin_targ,
            cos_targ,
            self.actual_l_thrust,
            self.actual_r_thrust,
            norm_angular_vel,
        ]
