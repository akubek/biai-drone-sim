import os
import argparse
import sys

from src.ai import neat_eval
from src.utils import test_physics, sim_runner

def parse_and_run() -> None:
    parser = argparse.ArgumentParser(description="BIAI Drone Sim - Symulator i Ewolucja NEAT")

    # ==========================================
    # 1. GŁÓWNY TRYB DZIAŁANIA (Wybór jednokrotny)
    # ==========================================
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train-fast", "train-visual", "showcase", "manual", "baseline"],
        default="train-visual",
        help="Główny tryb działania programu."
    )

    # ==========================================
    # 2. MODYFIKATORY (Argumenty pomocnicze)
    # ==========================================
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_drone_cascade.pkl",
        help="Ścieżka do pliku modelu. Używane w trybie 'showcase'."
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Ścieżka do pliku checkpointu (np. 'checkpoints/neat-checkpoint-50'). Używane w trybach 'train-*'."
    )

    parser.add_argument(
        "--arch",
        type=str,
        choices=["cascade", "e2e"],
        default="cascade",
        help="Architektura sterowania: 'cascade' (z FlightControllerem) lub 'e2e' (surowe silniki)."
    )

    args = parser.parse_args()

    # --- Ustalanie ścieżki do konfigu (zgodnie z nową strukturą) ---
    local_dir = os.path.dirname(__file__)
    if args.arch == "cascade":
        config_path = os.path.abspath(os.path.join(local_dir, "../conf/neat-cascade.txt"))
        is_cascade = True
    else:
        config_path = os.path.abspath(os.path.join(local_dir, "../conf/neat-e2e.txt"))
        is_cascade = False

    if not os.path.exists(config_path):
        print(f"❌ BŁĄD: Nie znaleziono pliku konfiguracyjnego NEAT: {config_path}")
        sys.exit(1)

    # ==========================================
    # 3. ROUTING LOGIKI
    # ==========================================
    if args.mode == "train-fast":
        print("🚀 TRYB: TRAIN-FAST (Headless, Wieloprocesowy)")
        if args.resume:
            print(f"Wznawianie z checkpointu: {args.resume}")
        # Tutaj wywołasz nową funkcję, która odpali ParallelEvaluatora bez Pygame
        neat_eval.run_neat_headless(config_path, checkpoint=args.resume, use_cascade=is_cascade)

    elif args.mode == "train-visual":
        print("👁️ TRYB: TRAIN-VISUAL (Z podglądem na żywo)")
        if args.resume:
            print(f"Wznawianie z checkpointu: {args.resume}")
        # To Twój dotychczasowy run_neat, przerobiony na oddzielne renderowanie
        neat_eval.run_neat_visual(config_path, checkpoint=args.resume, use_cascade=is_cascade)

    elif args.mode == "showcase":
        print(f"🎬 TRYB: SHOWCASE (Odtwarzanie: {args.model})")
        if not os.path.exists(args.model):
            print(f"❌ BŁĄD: Nie znaleziono pliku modelu '{args.model}'!")
            sys.exit(1)
        sim_runner.test_best_drone(config_path, genome_path=args.model)

    elif args.mode == "manual":
        print("🎮 TRYB: MANUAL (Test fizyki i sterownika z klawiatury)")
        test_physics.test_manual_flight()

    elif args.mode == "baseline":
        print("🤖 TRYB: BASELINE (Test wbudowanego Eksperta)")
        sim_runner.test_baseline()

def main() -> None:
    parse_and_run()

if __name__ == "__main__":
    main()