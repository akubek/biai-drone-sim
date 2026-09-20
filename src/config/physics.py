# --- SILNIK FIZYCZNY ---
GRAVITY = 9.81
THRUST_POWER = 0.4
DRAG = 0.5               # Opór powietrza (tłumienie prędkości liniowej)
ANGULAR_DRAG = 0.05        # Tłumienie prędkości obrotowej
TORQUE_POWER = 0.5        # Siła obrotu przy różnicy ciągów
TURN_SPEED = 4.0          # Mnożnik obrotu wizualnego/faktycznego
SAFE_CRASH_SPEED_M_S = 0.5 # Prędkość, którą uznajemy za bezpieczną w zderzeniu

# --- SENSORY I NORMALIZACJA ---
MAX_SPEED_NORM = 15.0     # Wartość do normalizacji prędkości (wejście sieci)
MAX_ANGULAR_NORM = 5.0    # Wartość do normalizacji prędkości kątowej (wejście sieci)
MAX_SENSOR_DIST_M = 2.5
RAYCAST_STEP_M = 0.1        # Skok promienia przy sprawdzaniu kolizji