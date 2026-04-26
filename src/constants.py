# src/constants.py

# --- USTAWIENIA OKNA I SYMULACJI ---
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 720
FPS = 60
SIMULATION_TIME = 10  # Maksymalny czas życia drona w sekundach
TARGET_SIZE = 50  # Promień punktu docelowego
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

# --- SENSORY ---
MAX_SENSOR_DIST = 250
RADAR_RANGE = 150
RAYCAST_STEP = 7.5  # Skok promienia przy sprawdzaniu kolizji

# --- PARAMETRY FITNESS (NAGRODY I KARY) ---
FIT_START_CAPITAL = 100.0

# Kary
FIT_CRASH_MULT = 0.5  # Zabiera 50% kapitału przy uderzeniu
FIT_CRASH_BASE_PENALTY = 20.0  # Dodatkowa kara bazowa za rozbicie
FIT_LAZY_PENALTY = 0.2  # Kara co klatkę za brak ruchu (lenistwo)
FIT_ESCAPE_PENALTY_MULT = 0.8  # Kara za oddalenie się od startu (na koniec)/
FIT_STAGNATION_MULT = 0.9  # Kara za brak postępu przez X klatek

# Nagrody i bonusy
FIT_DISCOVERY_BONUS = 500.0  # Jednorazowa nagroda za dotknięcie celu
FIT_SMOOTHNESS_MULT = 0.5  # Mnożnik nagrody za płynność lotu (brak szarpania)
FIT_SURVIVAL_FRAME_REWARD = 0.5
FIT_JITTER_PENALTY = 0.2
FIT_STABILITY_MULT = (
    0.2  # Nagroda za brak niepotrzebnych obrotów (kątowych) w pobliżu celu
)
FIT_EXPLORATION_MULT = 1.0  # Mnożnik za każdy piksel przybliżenia się do celu
FIT_HOVER_REWARD = 5.0  # Nagroda bazowa co klatkę za utrzymanie się w strefie celu
FIT_HOVER_SUCCESS_BONUS = (
    2000.0  # Duża nagroda za ustabilizowanie się na celu przez 1 sekundę
)
FIT_SURVIVAL_BONUS_MULT = 5.0  # Bonus na koniec za zbliżenie się i przetrwanie
FIT_SPIN_PENALTY = 0.5  # Kara (mnożnik), jeśli dron przesadzi z kręceniem
MAX_ALLOWED_SPINS = 2.0  # Maksymalny obrót w jedną stronę (2 pełne pętle)

STAGNATION_LIMIT_SEC = 2.5  # Limit czasu bez postępu
HOVER_REQUIRED_SEC = 1.5  # Wymagany czas by uznać zadanie za wykonane

EVOLUTION_CYCLES = 500
