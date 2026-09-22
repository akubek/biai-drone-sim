# --- WAGI SKLADNIKOW FITNESSU ---
# Wszystkie skladniki sa bezwymiarowe (ulamki), wiec te liczby to jawne wagi
# i mozna je porownywac miedzy soba. Zakres calosci: ok. -0.6 .. 4.5,
# niezaleznie od mapy, dystansu startowego i etapu trudnosci.
FIT_W_PROGRESS = 1.0      # za zamkniety ulamek dystansu do celu
FIT_W_DISCOVERY = 0.5     # jednorazowo za dotkniecie strefy celu
FIT_W_HOVER = 1.0         # za ulamek wymaganego czasu zawisu
FIT_W_SUCCESS = 2.0       # za ukonczenie zadania
FIT_CRASH_FORFEIT = 0.5         # kara za kolizje (ułamek zdobytych punktów ktory zostaje odebrany)
FIT_KAMIKAZE_FORFEIT = 0.8      # dodatkowa kara za uderzenie z predkoscia (ułamek zdobytych punktów ktory zostaje odebrany)
FIT_ESCAPE_FORFEIT = 0.5       # kara za ucieczke (ułamek zdobytych punktów ktory zostaje odebrany)
FIT_W_ENERGY = 0.0        # kara za zuzycie energii
FIT_W_BRAKING = 2.0        # nagroda za hamowanie (ułamek zdobytych punktów ktory zostaje przyznany)

assert 0.0 <= FIT_CRASH_FORFEIT < FIT_KAMIKAZE_FORFEIT < 1.0