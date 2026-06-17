# --- KAPITAŁ STARTOWY ---
FIT_START_CAPITAL = 0.0

# --- KARY (Punktowe) ---
FIT_CRASH_BASE_PENALTY = 10.0   # Płaska kara za uderzenie w przeszkodę
FIT_KAMIKAZE_PENALTY = 15.0     # Dodatkowa kara za uderzenie bez hamowania
FIT_EXPERT_PENALTY_MULT = 10.0  # Kara za ignorowanie eksperta (Imitation Learning)

# --- NAGRODY I BONUSY ---
FIT_DISCOVERY_BONUS = 50.0      # Jednorazowa nagroda za dotknięcie celu
FIT_EXPLORATION_MULT = 10.0     # Mnożnik za bicie rekordów dystansu (ok. 60 pkt za mapę)
FIT_HOVER_REWARD = 200.0        # Nagroda za utrzymanie się w strefie celu [pkt/sek]
FIT_HOVER_SUCCESS_REWARD = 1000.0 # Płaska nagroda za ukończenie zadania