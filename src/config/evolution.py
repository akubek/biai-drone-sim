# --- ZASADY SYMULACJI (Early Stopping) ---
import math

SIMULATION_TIME = 15            # Max czas życia drona w sek. (jeśli nie utknie)
HOVER_REQUIRED_SEC = 1.5        # Wymagany czas w celu by uznać zadanie za wykonane
HOVER_MAX_SPEED_M_S = 0.3       # Maksymalna predkosc drona podczas zawisu
HOVER_MAX_ANGULAR_VEL = 1.0     # Maksymalna predkosc kątowa drona podczas zawisu
V_REF = 5.0
W_REF = 5.0

# --- RESTRYKCJE I STAGNACJA ---
STAGNATION_LIMIT_SEC = 5.0      # Zwiększony limit czasu bez postępu
FIT_STAGNATION_DISTANCE_LIMIT_M = 0.2 # Min. postęp w metrach by zresetować zegar stagnacji
ESCAPE_LIMIT_PERC = 0.3         # 30% przekątnej mapy
MAX_SAFE_ANGULAR_VEL = 4.0 * math.pi
MAX_ALLOWED_SPINOUT_TIME = 0.25

# --- HARMONOGRAM TRENINGU ---
EVOLUTION_CYCLES = 200