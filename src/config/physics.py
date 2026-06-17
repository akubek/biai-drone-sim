# --- SILNIK FIZYCZNY ---
GRAVITY = 9.98
THRUST_POWER = 0.4
DRAG = 0.98               # Opór powietrza (tłumienie prędkości liniowej)
ANGULAR_DRAG = 0.9        # Tłumienie prędkości obrotowej
TORQUE_POWER = 0.5        # Siła obrotu przy różnicy ciągów
TURN_SPEED = 4.0          # Mnożnik obrotu wizualnego/faktycznego
SAFE_CRASH_SPEED_M_S = 0.5 # Prędkość, którą uznajemy za bezpieczną w zderzeniu

# --- SENSORY I NORMALIZACJA ---
MAX_SPEED_NORM = 15.0     # Wartość do normalizacji prędkości (wejście sieci)
MAX_ANGULAR_NORM = 5.0    # Wartość do normalizacji prędkości kątowej (wejście sieci)
MAX_SENSOR_DIST = 250
RADAR_RANGE = 250
RAYCAST_STEP = 7.5        # Skok promienia przy sprawdzaniu kolizji