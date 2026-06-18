import json
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
        help="Program mode."
    )

    # ==========================================
    # 2. MODIFIERS (Auxiliary arguments)
    # ==========================================
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_drone_cascade.pkl",
        help="Path to the model file. Used in 'showcase' mode."
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to the checkpoint file (e.g., 'checkpoints/neat-checkpoint-50'). Used in 'train-*' modes."
    )

    parser.add_argument(
        "--arch",
        type=str,
        choices=["cascade", "e2e"],
        default="cascade",
        help="Control architecture: 'cascade' (with FlightController) or 'e2e' (raw motors)."
    )

    args = parser.parse_args()

    # --- LOADING EXPERIMENT CONFIGURATION FROM JSON ---
    config_json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "training_config.json"))
    try:
        with open(config_json_path, 'r', encoding='utf-8') as f:
            exp_config = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Experiment configuration file not found: {config_json_path}")
        sys.exit(1)

    training_mode = exp_config.get("training_mode", 0)
    target_obstacles = exp_config.get("target_obstacles", 3)

    # --- Determine the path to the config file (according to the new structure) ---
    local_dir = os.path.dirname(__file__)
    if args.arch == "cascade":
        config_path = os.path.abspath(os.path.join(local_dir, "../conf/neat-cascade.txt"))
        is_cascade = True
    else:
        config_path = os.path.abspath(os.path.join(local_dir, "../conf/neat-e2e.txt"))
        is_cascade = False

    if not os.path.exists(config_path):
        print(f"ERROR: NEAT config file not found: {config_path}")
        sys.exit(1)

    # ==========================================
    # 3. ROUTING LOGIC
    # ==========================================
    if args.mode == "train-fast":
        print("MODE: TRAIN-FAST (Headless, Multi-process)")
        if args.resume:
            print(f"Resuming from checkpoint: {args.resume}")
        # Here you will call the new function that runs the ParallelEvaluator without Pygame
        neat_eval.run_neat_headless(
            config_path,
            checkpoint=args.resume,
            use_cascade=is_cascade,
            exp_config=exp_config
        )

    elif args.mode == "train-visual":
        print("MODE: TRAIN-VISUAL (With live preview)")
        if args.resume:
            print(f"Resuming from checkpoint: {args.resume}")
        # This is your previous run_neat, modified for separate rendering
        neat_eval.run_neat_visual(
            config_path,
            checkpoint=args.resume,
            use_cascade=is_cascade,
            exp_config=exp_config
        )

    elif args.mode == "showcase":
        print(f"MODE: SHOWCASE (Playback: {args.model})")
        if not os.path.exists(args.model):
            print(f"ERROR: Model file not found: '{args.model}'")
            sys.exit(1)
        sim_runner.test_best_drone(config_path, genome_path=args.model)

    elif args.mode == "manual":
        print("MODE: MANUAL (Test physics and controller with keyboard)")
        test_physics.test_manual_flight()

    elif args.mode == "baseline":
        print("MODE: BASELINE (Test built-in Expert)")
        sim_runner.test_baseline()

def main() -> None:
    parse_and_run()

if __name__ == "__main__":
    main()