import os
import sys

from src import evolution


def main() -> None:
    print("🚀 Inicjalizacja środowiska BIAI...")

    # Ponieważ main.py jest w folderze 'src', musimy wskazać plik config jeden folder wyżej ('..')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "..", "config-feedforward.txt")

    # Zabezpieczenie: sprawdzamy, czy plik na pewno tam jest
    if not os.path.exists(config_path):
        print(
            f"❌ BŁĄD: Nie znaleziono pliku konfiguracyjnego pod adresem:\n{config_path}"
        )
        print(
            "Upewnij się, że plik 'config-feedforward.txt' znajduje się w głównym folderze projektu."
        )
        sys.exit(1)

    # Odpalamy "mózg" projektu!
    print("🧠 Uruchamianie algorytmu genetycznego NEAT...")
    evolution.run_neat(config_path)


if __name__ == "__main__":
    main()
