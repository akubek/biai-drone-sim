"""Generuje zamrozony zbior holdout dla wybranej drabinki trudnosci.

Plik jest zamrozony - regeneracja uniewaznia porownania z wczesniejszymi
przebiegami na TEJ SAMEJ drabince.
"""
import argparse
import json
import random
from pathlib import Path

from src.core.environment import TIERS, generate_scenario_for_tier, select_ladder

HOLDOUT_SEED = 987654321
PER_TIER = 20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ladder", choices=["v1", "v2"], default="v1")
    parser.add_argument("--per-tier", type=int, default=PER_TIER)
    parser.add_argument("--force", action="store_true",
                        help="Nadpisz istniejacy plik.")
    args = parser.parse_args()

    select_ladder(args.ladder)
    out = Path(f"data/holdout_{args.ladder}.json")

    if out.exists() and not args.force:
        print(f"{out} juz istnieje - uzyj --force, zeby nadpisac. "
              f"Uniewazni to porownania z wczesniejszymi przebiegami.")
        return

    random.seed(HOLDOUT_SEED)
    scenarios = [generate_scenario_for_tier(t)
                 for t in TIERS for _ in range(args.per_tier)]

    blob = {
        "version": 2,
        "ladder": args.ladder,
        "seed": HOLDOUT_SEED,
        "per_tier": args.per_tier,
        "tiers": {str(k): v["description"] for k, v in TIERS.items()},
        "scenarios": [
            {
                "tier": s.tier,
                "start_px": list(s.start_px),
                "target_px": list(s.target_px),
                "obstacles_px": [list(o) for o in s.obstacles_px],
            }
            for s in scenarios
        ],
    }
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    print(f"zapisano {len(scenarios)} scenariuszy do {out}")


if __name__ == "__main__":
    main()