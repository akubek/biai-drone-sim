import argparse
import json
import os
import random
import sys

from src.ai import neat_eval
from src.core.environment import select_ladder
from src.utils import manual_flight, sim_runner


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
        help="Path to the checkpoint file (e.g., 'checkpoints/neat-checkpoint-50') or if none latest is used. Used in 'train-*' modes. "
             "Training resumed from a checkpoint is no longer considered reproducible."
    )

    parser.add_argument(
        "--arch",
        type=str,
        choices=["cascade", "e2e"],
        default="cascade",
        help="Control architecture: 'cascade' (with FlightController) or 'e2e' (raw motors)."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for random number generator, used for reproducibility."
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Number of generations for training in 'train-*' modes."
    )

    parser.add_argument(
        "--pop-size",
        type=int,
        default=None,
        help="Population size for training in 'train-*' modes."
    )

    parser.add_argument(
        "--net-type",
        type=str,
        choices=["feedforward", "recurrent"],
        default="feedforward",
        help="Network type. Sets the feed_forward parameter in the NEAT configuration "
            "(overrides the value from the conf/ file)."
    )

    parser.add_argument(
        "--scenarios",
        type=int,
        default=None,
        help="Number of scenarios (maps) per genome in each generation."
    )

    parser.add_argument(
        "--start-tier",
        type=int,
        default=None,
        help="Starting tier for the network. Used in 'train-*' modes."
    )

    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Disable curriculum learning, learning only at starting tier."
    )

    parser.add_argument(
        "--scenario-refresh",
        type=int,
        default=20,
        help="Number of generations after which scenarios are refreshed."
    )
    parser.add_argument("--ladder",
        choices=["v1", "v2"],
        default="v1",
        help="Variant of the difficulty ladder.",
        )

    parser.add_argument(
        "--holdout-every",
        type=int,
        default=None,
        help="Number of generations after which holdout scenarios are evaluated."
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

    exp_config["arch"] = args.arch
    #set rng seed from argument or generate a random one if not provided
    seed = args.seed if args.seed is not None else random.randrange(2**31)
    random.seed(seed)
    exp_config["seed"] = seed
    print(f"SEED: {seed}" )
    exp_config["net_type"] = args.net_type
    if args.start_tier is not None:
        exp_config["start_tier"] = args.start_tier
    if args.no_curriculum:
        exp_config["curriculum"] = False

    exp_config["ladder"] = args.ladder
    select_ladder(exp_config.get("ladder", "v1"))

    exp_config.setdefault("start_tier", 1)
    exp_config.setdefault("curriculum", True)

    print(f"TIER: {exp_config['start_tier']} | CURRICULUM: {exp_config['curriculum']}")

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

    if args.generations is not None:
        exp_config["generations"] = args.generations
    if args.scenarios is not None:
        exp_config["scenarios_per_genome"] = args.scenarios
    if args.pop_size is not None:
        exp_config["pop_size"] = args.pop_size
    if args.holdout_every is not None:
        exp_config["holdout_every"] = args.holdout_every
    if args.scenario_refresh is not None:
        exp_config["scenario_refresh"] = args.scenario_refresh

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
        manual_flight.run_manual_flight()

    elif args.mode == "baseline":
        print("MODE: BASELINE (Test built-in Expert)")
        sim_runner.test_baseline()

def main() -> None:
    parse_and_run()

if __name__ == "__main__":
    main()