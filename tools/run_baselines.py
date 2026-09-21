"""Measures baselines (upper and lower bounds) on the holdout set: expert (ceiling) and unevolved networks (floor).

Baselines on the holdout set: expert (ceiling) and unevolved networks (floor).

Without these two numbers, the agent's performance has no scale - it's unclear whether 12% success
is a lot or just random chance.

It also checks whether the difficulty levels are actually ordered: the expert's success rate
should decrease from level 1 upwards. If not - adjust the TIERS,
regenerate the holdout set and repeat, BEFORE freezing the set.

Usage (from the project directory):
    python -m tools.run_baselines
    python -m tools.run_baselines --arch e2e --genomes 30
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import itertools

import neat

from src.ai.expert import HardcodedBrain
from src.ai.neat_eval import _apply_net_type, _run_episode
from src.core.environment import load_holdout
from src.core.stats import EndReason, EpisodeResult

# help_weight=1.0 -> step_training_drone takes ONLY the expert's output,
# help_weight=0.0 -> only the network's output. The same simulation loop is used for both.
EXPERT_WEIGHT = 1.0
NETWORK_WEIGHT = 0.0


@dataclass
class TierStats:
    tier: int
    n: int
    success_rate: float
    crash_rate: float
    escape_rate: float
    mean_dist_ratio: float
    mean_time_to_target: float | None


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _summarize(results: list[EpisodeResult], tier: int) -> TierStats:
    n = len(results)
    successes = [r for r in results if r.success]
    return TierStats(
        tier=tier,
        n=n,
        success_rate=len(successes) / n,
        crash_rate=sum(1 for r in results if r.end_reason is EndReason.CRASH) / n,
        escape_rate=sum(1 for r in results if r.end_reason is EndReason.ESCAPE) / n,
        mean_dist_ratio=statistics.fmean(r.min_dist_ratio for r in results),
        mean_time_to_target=(statistics.fmean(r.time_alive_s for r in successes)
                             if successes else None),
    )


def evaluate(genomes, config, scenarios, help_weight: float) -> list[TierStats]:
    """Each genome runs through every scenario. Results are grouped by tier."""
    expert = HardcodedBrain()
    by_tier: dict[int, list[EpisodeResult]] = defaultdict(list)

    for scenario in scenarios:
        for genome in genomes:
            by_tier[scenario.tier].append(
                _run_episode(genome, config, scenario, expert, help_weight)
            )

    return [_summarize(by_tier[t], t) for t in sorted(by_tier)]


def print_table(title: str, stats: list[TierStats]) -> None:
    print(f"\n{title}")
    print(f"{'level':>7} {'n':>6} {'success':>8} {'crash':>9} {'escape':>9} "
          f"{'dist_ratio':>11} {'time to target':>13}")
    for s in stats:
        czas = f"{s.mean_time_to_target:.2f}" if s.mean_time_to_target is not None else "-"
        print(f"{s.tier:>7} {s.n:>6} {s.success_rate:>8.3f} {s.crash_rate:>9.3f} "
              f"{s.escape_rate:>9.3f} {s.mean_dist_ratio:>11.3f} {czas:>13}")


def check_monotonic(stats: list[TierStats]) -> list[str]:
    """The expert's success rate should decrease, and the crash rate should increase with the tier."""
    problems = []
    for prev, curr in itertools.pairwise(stats):
        if curr.success_rate > prev.success_rate + 0.05:
            problems.append(
                f"level {curr.tier} easier than {prev.tier} "
                f"(success {curr.success_rate:.3f} > {prev.success_rate:.3f})"
            )
        if curr.crash_rate < prev.crash_rate - 0.05:
            problems.append(
                f"level {curr.tier} less crash-prone than {prev.tier} "
                f"({curr.crash_rate:.3f} < {prev.crash_rate:.3f})"
            )
    return problems


def _md_table(stats: list[TierStats]) -> str:
    lines = ["| level | n | success | crash | escape | dist_ratio | time to target [s] |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for s in stats:
        czas = f"{s.mean_time_to_target:.2f}" if s.mean_time_to_target is not None else "–"
        lines.append(f"| {s.tier} | {s.n} | {s.success_rate:.3f} | {s.crash_rate:.3f} | "
                     f"{s.escape_rate:.3f} | {s.mean_dist_ratio:.3f} | {czas} |")
    return "\n".join(lines)


def write_markdown(path: Path, expert: list[TierStats], random_nets: list[TierStats],
                   meta: dict, problems: list[str]) -> None:
    def overall(stats):
        total = sum(s.n for s in stats)
        return sum(s.success_rate * s.n for s in stats) / total if total else 0.0

    parts = [
        "# Baselines on the holdout set",
        "",
        "Reference points for the agent's performance. **Ceiling** is the hand-coded controller",
        "(`HardcodedBrain`), **floor** is the untrained networks with random weights,",
        "i.e., what is seen in the zero-th generation.",
        "",
        "| parameter | value |",
        "|---|---|",
    ]
    parts += [f"| {k} | {v} |" for k, v in meta.items()]
    parts += [
        "",
        f"**Overall success:** expert {overall(expert):.3f} | random networks {overall(random_nets):.3f}",
        "",
        "## Expert (ceiling)",
        "",
        _md_table(expert),
        "",
        "## Untrained networks (floor)",
        "",
        _md_table(random_nets),
        "",
    ]
    if problems:
        parts += ["## WARNING: unordered tiers", ""]
        parts += [f"- {p}" for p in problems]
        parts += ["", "Fix `TIERS`, regenerate `data/holdout.json` and repeat the measurement.", ""]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Expert and random networks baselines.")
    parser.add_argument("--arch", choices=["cascade", "e2e"], default="cascade")
    parser.add_argument("--net-type", choices=["feedforward", "recurrent"], default="feedforward")
    parser.add_argument("--genomes", type=int, default=20,
                        help="Number of random genomes for the floor (averaged).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout", default="data/holdout.json")
    parser.add_argument("--out", default="docs/baselines.md")
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    config_path = f"conf/neat-{args.arch}.txt"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    _apply_net_type(config, args.net_type)
    cast(Any, config).use_cascade = (args.arch == "cascade")

    scenarios = load_holdout(args.holdout)
    print(f"holdout: {len(scenarios)} scenarios, "
          f"{len({s.tier for s in scenarios})} tiers")

    # Populacja tylko po to, zeby dostac poprawnie zainicjalizowane genomy
    # (konstruktor ustawia m.in. innovation_tracker).
    population = neat.Population(config)
    all_genomes = list(population.population.values())
    random_genomes = all_genomes[:args.genomes]

    print(f"ekspert: {len(scenarios)} epizodow")
    expert_stats = evaluate(all_genomes[:1], config, scenarios, EXPERT_WEIGHT)

    print(f"losowe sieci: {len(scenarios) * len(random_genomes)} epizodow")
    random_stats = evaluate(random_genomes, config, scenarios, NETWORK_WEIGHT)

    print_table("EKSPERT (sufit)", expert_stats)
    print_table("NIEEWOLUOWANE SIECI (podloga)", random_stats)

    problems = check_monotonic(expert_stats)
    if problems:
        print("\nUWAGA - poziomy nie sa uporzadkowane wg trudnosci:")
        for p in problems:
            print(f"  - {p}")
        print("  Popraw TIERS, zregeneruj holdout i powtorz.")
    else:
        print("\nPoziomy uporzadkowane poprawnie (skutecznosc eksperta spada).")

    meta = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "commit": _git_commit(),
        "arch": args.arch,
        "net_type": args.net_type,
        "seed": args.seed,
        "holdout": args.holdout,
        "scenarios": len(scenarios),
        "genomes (floor)": len(random_genomes),
    }
    write_markdown(Path(args.out), expert_stats, random_stats, meta, problems)
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()