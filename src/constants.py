# src/constants.py

# --- USTAWIENIA OKNA I SYMULACJI ---
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 720
FPS = 60
SIMULATION_TIME = 10  # Maksymalny czas życia drona w sekundach
TARGET_SIZE_PX = 50  # Promień punktu docelowego
PPM = 200.0
# --- FIZYKA DRONA ---
GRAVITY = 0.15
THRUST_POWER = 0.4
DRAG = 0.98  # Opór powietrza (tłumienie prędkości liniowej)
ANGULAR_DRAG = 0.9  # Tłumienie prędkości obrotowej
TORQUE_POWER = 0.5  # Siła obrotu przy różnicy ciągów
TURN_SPEED = 4.0  # Mnożnik obrotu wizualnego/faktycznego
MAX_SPEED_NORM = 15.0  # Wartość do normalizacji prędkości (wejście sieci)
MAX_ANGULAR_NORM = 5.0  # Wartość do normalizacji prędkości kątowej (wejście sieci)

# --- PARAMETRY MAPY I GENERACJI ---
MAP_MARGIN_PX = 80  # Minimalna odległość drona/celu od krawędzi ekranu
MIN_SPAWN_DISTANCE_PX = (
    500.0  # Minimalna odległość w linii prostej między startem a celem
)

# --- SENSORY ---
MAX_SENSOR_DIST = 250
RADAR_RANGE = 150
RAYCAST_STEP = 7.5  # Skok promienia przy sprawdzaniu kolizji

# --- PARAMETRY FITNESS (NAGRODY I KARY) ---
FIT_START_CAPITAL = 100.0

# Kary
SAFE_CRASH_SPEED_M_S = 0.5  # prędkość którą uznajemy za bezpieczną w zderzeniu
FIT_CRASH_PENALTY_PERC = 0.75  # Zabiera ułamek kapitału przy uderzeniu
FIT_CRASH_BASE_PENALTY = 5.0  # Kara punktowa - bazowa za rozbicie
FIT_KAMIKAZE_PENALTY = 20.0
FIT_LAZY_PENALTY = 1.0  # Kara punktowa co klatkę za brak ruchu (lenistwo)
FIT_ESCAPE_PENALTY_PERC = (
    0.75  # Zabiera ułamek kapitału - kara za oddalenie się od startu (na koniec) -
)
FIT_STAGNATION_PENALTY_PERC = (
    0.25  #  Kara procentowa na koniec za brak postępu po X sekund
)
FIT_STAGNATION_DISTANCE_LIMIT_M = 0.5  # limit poprawy wyniku odległości w [m] poniżej którego stwierdza że dron się nie poprawia
SURVIVAL_TIME_PEAK = 2.5  # W której sekundzie nagroda jest największa
FIT_SURVIVAL_FRAME_REWARD = (
    2.0  # nagroda za przetrwanie - ile drony dostaja za przerzycie na początku
)

ESCAPE_LIMIT = 2.0  # limit ile moze oldeciec od celu wzgledem poczatkowej pozycji

FIT_IDLE_PENALTY = 5.0  # punktowo penalty za brak ruchu - idle [punkty/sek]

IDLE_MIN_SPEED = 0.1  # minimalna predkość drona poniżej ktorej uznajemy go za idle,
# Nagrody i bonusy
FIT_DISCOVERY_BONUS = 500.0  # Jednorazowa nagroda za dotknięcie celu
FIT_SMOOTHNESS_MULT = 0.5  # Mnożnik nagrody za płynność lotu (brak szarpania)
FIT_JITTER_PENALTY = 0.2
FIT_STABILITY_MULT = (
    0.2  # Nagroda za brak niepotrzebnych obrotów (kątowych) w pobliżu celu
)
FIT_EXPLORATION_MULT = (
    500.0  # Multiplier za odległość o którą dron przybliżył się do celu [pkt/m]
)

FIT_DIR_VELOCITY_REWARD = 2.0  # Punkty za każdy 1 m/s lotu IDEALNIE w stronę celu
FIT_WRONG_DIR_PENALTY = 1.0  # Kara za każdy 1 m/s dryfowania W PRZECIWNĄ stronę

FIT_HOVER_REWARD = 500.0  # Nagroda za utrzymanie się w strefie celu [pkt/sek]
FIT_HOVER_SUCCESS_BONUS = (
    2500.0  # Duża nagroda za ustabilizowanie się na celu przez 1 sekundę
)
FIT_SURVIVAL_BONUS_MULT = (
    1.5  # Bonus multiplier na koniec za zbliżenie się i przetrwanie
)
FIT_SPIN_PENALTY = 50.0  # Kara co klatkę, jeśli dron przesadzi z kręceniem
MAX_ALLOWED_SPINS = 3.0  # Maksymalny obrót w jedną stronę po którym zaczynamy nabijać karę za kręcenie (2 pełne pętle)

IDLE_LIMIT_SEC = 1.0
STAGNATION_LIMIT_SEC = 2.5  # Limit czasu bez postępu w sekundach
HOVER_REQUIRED_SEC = 1.5  # Wymagany czas by uznać zadanie za wykonane

EVOLUTION_CYCLES = 500
