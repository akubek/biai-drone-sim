# --- WAGI SKLADNIKOW FITNESSU ---
# Wszystkie skladniki sa bezwymiarowe (ulamki), wiec te liczby to jawne wagi
# i mozna je porownywac miedzy soba. Zakres calosci: ok. -0.6 .. 4.5,
# niezaleznie od mapy, dystansu startowego i etapu trudnosci.
FIT_W_PROGRESS = 1.0      # za zamkniety ulamek dystansu do celu
FIT_W_DISCOVERY = 0.5     # jednorazowo za dotkniecie strefy celu
FIT_W_HOVER = 1.0         # za ulamek wymaganego czasu zawisu
FIT_W_SUCCESS = 2.0       # za ukonczenie zadania
FIT_W_CRASH = 0.3         # kara za kolizje
FIT_W_KAMIKAZE = 0.3      # dodatkowa kara za uderzenie z predkoscia
FIT_W_ENERGY = 0.0        # wlaczane w #21