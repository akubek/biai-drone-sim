import os
import argparse
import sys

from src import evolution
from src import test_physics


def parse_and_run() -> None:
    parser = argparse.ArgumentParser(description="BIAI Drone Sim - Ewolucja NEAT")

    # Dodajemy flagę --replay-best (jeśli podana, wartość to True, domyślnie False)
    parser.add_argument(
        "--replay-best",
        action="store_true",
        help="Odtwórz najlepszego drona zamiast rozpoczynać nowy trening.",
    )

    # Dodajemy opcjonalny argument na nazwę pliku (przydatne, jak masz kilka modeli)
    parser.add_argument(
        "--model",
        type=str,
        default="best_drone.pkl",
        help="Ścieżka do pliku z zapisanym modelem (domyślnie: best_drone.pkl)",
    )

    parser.add_argument(
        "--resume",
        type=str,
        nargs="?",
        const="latest",
        help="Wznów trening z checkpointu. Możesz podać nazwę pliku lub zostawić puste dla najnowszego.",
    )

    parser.add_argument(
        "--test-baseline",
        action="store_true",
        help="Wznów trening z checkpointu. Możesz podać nazwę pliku lub zostawić puste dla najnowszego.",
    )

    parser.add_argument(
        "--manual",
        action="store_true",
        help="Wznów trening z checkpointu. Możesz podać nazwę pliku lub zostawić puste dla najnowszego.",
    )

    # Odczytujemy to, co użytkownik wpisał w terminalu
    args = parser.parse_args()

    # --- 2. Ścieżki do plików ---
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "../config-feedforward.txt")

    # --- 3. Logika wyboru trybu ---
    if args.replay_best:
        print(f"TRYB POKAZOWY: Uruchamianie zapisanego drona z pliku '{args.model}'...")

        # Zabezpieczenie: sprawdzamy czy plik w ogóle istnieje
        if not os.path.exists(args.model):
            print(f"❌ BŁĄD: Nie znaleziono pliku '{args.model}'!")
            print(
                "Najpierw musisz wytrenować drona uruchamiając program bez flagi --replay-best."
            )
            sys.exit(1)

        evolution.test_best_drone(config_path, genome_path=args.model)
    elif args.resume:
        # Logika wznawiania
        checkpoint_file = args.resume
        if checkpoint_file == "latest":
            # Tu można by dodać logikę szukania pliku z najwyższym numerem,
            # ale na razie załóżmy, że podajesz nazwę lub używasz domyślnej
            print("Wznawianie z najnowszego dostępnego checkpointu...")

        evolution.run_neat(config_path, checkpoint=checkpoint_file)
    elif args.manual:
        test_physics.test_manual_flight()
    elif args.test_baseline:
        evolution.test_baseline()
    else:
        print("TRYB TRENINGU: Rozpoczynanie ewolucji NEAT...")
        evolution.run_neat(config_path)


def main() -> None:
    parse_and_run()
    # test_physics.test_manual_flight()
    return


if __name__ == "__main__":
    main()
