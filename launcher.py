import os
import subprocess
import sys


def run_command(command: list[str], description: str):
    """Pomocnicza funkcja do uruchamiania komend z obsługą błędów."""
    print(f"{description}...")
    try:
        # check=True sprawia, że przy błędzie rzucony zostanie wyjątek
        # capture_output=False pozwala nam widzieć postęp instalacji w terminalu
        _ = subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: {description} failed!")
        print(f"Command: {' '.join(command)}")
        print(f"Exit code: {e.returncode}")
        sys.exit(1)  # Zamyka skrypt, bo nie ma sensu iść dalej
    except FileNotFoundError:
        print(f"\n❌ ERROR: Could not find the command to run: {command[0]}")
        sys.exit(1)


def main():
    print("--- BIAI Drone Sim Launcher ---")

    # 1. Tworzenie venv
    if not os.path.exists("venv"):
        run_command(
            [sys.executable, "-m", "venv", "venv"], "Creating virtual environment"
        )

    # 2. Wybór ścieżki (Linux/Windows)
    if os.name == "nt":
        python_exe = os.path.join("venv", "Scripts", "python.exe")
    else:
        python_exe = os.path.join("venv", "bin", "python")

    # 3. Instalacja zależności
    if os.path.exists("requirements.txt"):
        run_command(
            [python_exe, "-m", "pip", "install", "-r", "requirements.txt"],
            "Installing dependencies",
        )
    else:
        print("Warning: requirements.txt not found. Skipping installation.")

    # 4. Uruchomienie gry
    print("Starting simulation...")
    try:
        # Tutaj nie używamy run_command, bo chcemy, żeby gra działała normalnie
        _ = subprocess.run([python_exe, "-m", "src.main"], check=True)
    except subprocess.CalledProcessError:
        print("\nSimulation closed with an error or interrupted.")
    except KeyboardInterrupt:
        print("\nSimulation stopped by user (Ctrl+C).")


if __name__ == "__main__":
    main()
