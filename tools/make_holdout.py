"""Generates a holdout set and saves it to data/holdout.json.

Run once. File is frozen - regenerating invalidates comparisons with all previous runs.
"""
import json
import random
from pathlib import Path

from src.core.environment import TIERS, generate_scenario_for_tier

HOLDOUT_SEED = 987654321      # Seed for reproducibility
PER_TIER = 20


def main() -> None:
    out = Path("data/holdout.json")
    if out.exists():
        print(f"{out} already exists - delete manually if you really want to regenerate. Will invalidate comparisons with previous runs.")
        return

    random.seed(HOLDOUT_SEED)
    scenarios = [generate_scenario_for_tier(t) for t in TIERS for _ in range(PER_TIER)]

    blob = {
        "version": 1,
        "seed": HOLDOUT_SEED,
        "per_tier": PER_TIER,
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
    print(f"saved {len(scenarios)} scenarios to {out}")


if __name__ == "__main__":
    main()