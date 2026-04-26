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
FIT_START_CAPITAL = 0.0

# Kary
SAFE_CRASH_SPEED_M_S = 0.5  # prędkość którą uznajemy za bezpieczną w zderzeniu
FIT_CRASH_PENALTY_PERC = 0.9  # Zabiera ułamek kapitału przy uderzeniu
FIT_CRASH_BASE_PENALTY = 5.0  # Kara punktowa - bazowa za rozbicie
FIT_KAMIKAZE_PENALTY = 10.0
FIT_LAZY_PENALTY = 1.0  # Kara punktowa co klatkę za brak ruchu (lenistwo)
FIT_ESCAPE_PENALTY_PERC = (
    0.75  # Zabiera ułamek kapitału - kara za oddalenie się od startu (na koniec) -
)

FIT_EXPERT_PENALTY_MULT = 10.0

FIT_STAGNATION_PENALTY_PERC = (
    0.25  #  Kara procentowa na koniec za brak postępu po X sekund
)
FIT_STAGNATION_DISTANCE_LIMIT_M = 0.25  # limit poprawy wyniku odległości w [m] poniżej którego stwierdza że dron się nie poprawia

ESCAPE_LIMIT = 2.0  # limit ile moze oldeciec od celu wzgledem poczatkowej pozycji

# Nagrody i bonusy
FIT_DISCOVERY_BONUS = 1000.0  # Jednorazowa nagroda za dotknięcie celu
FIT_EXPLORATION_MULT = (
    500.0  # Multiplier za odległość o którą dron przybliżył się do celu [pkt/m]
)

FIT_HOVER_REWARD = 2000.0  # Nagroda za utrzymanie się w strefie celu [pkt/sek]
FIT_HOVER_SUCCESS_BONUS = (
    5000.0  # Duża nagroda za ustabilizowanie się na celu przez 1 sekundę
)

IDLE_LIMIT_SEC = 1.0
STAGNATION_LIMIT_SEC = 2.5  # Limit czasu bez postępu w sekundach
HOVER_REQUIRED_SEC = 1.5  # Wymagany czas by uznać zadanie za wykonane

EVOLUTION_CYCLES = 500
