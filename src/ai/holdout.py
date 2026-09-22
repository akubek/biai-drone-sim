import csv
import os
import statistics
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from neat.reporting import BaseReporter

from src.core.environment import Scenario
from src.core.stats import EndReason, EpisodeResult

HEADERS = ["generation", "tier", "n", "success_rate", "mean_dist_ratio", "crash_rate"]


class HoldoutReporter(BaseReporter):
    """Every N generations evaluates the best genome on a fixed set of maps.

    The set is NEVER used in training - it measures generalization,
    not fitting to the maps on which the population was selected.
    """

    def __init__(self, scenarios: list[Scenario], run_episode_fn: Callable,
                 config: Any, folder: str, every: int = 10):
        self.scenarios = scenarios
        self.run_episode = run_episode_fn
        self.config = config
        self.every = every
        self.generation = 0
        self.last_overall_success = ""      # read by CSVTrainingReporter
        self.filename = os.path.join(folder, "holdout_log.csv")
        self.last_by_tier: dict[int, dict] = {} 
        with open(self.filename, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HEADERS)

    def start_generation(self, generation: int) -> None:
        self.generation = generation
        self.last_overall_success = ""
        self.last_by_tier = {}

    def post_evaluate(self, config, population, species, best_genome) -> None:
        if best_genome is None or self.generation % self.every != 0:
            return

        by_tier: dict[int, list[EpisodeResult]] = defaultdict(list)
        for scenario in self.scenarios:
            # help_weight=0.0: holdout measures the network alone, without expert support.
            result = self.run_episode(best_genome, self.config, scenario, None, 0.0)
            by_tier[scenario.tier].append(result)

        rows = []
        for tier in sorted(by_tier):
            results = by_tier[tier]
            n = len(results)
            succ = round(sum(1 for r in results if r.success) / n, 4)
            crash = round(sum(1 for r in results
                            if r.end_reason is EndReason.CRASH) / n, 4)
            dist = round(statistics.fmean(r.min_dist_ratio for r in results), 4)

            rows.append([self.generation, tier, n, succ, dist, crash])
            self.last_by_tier[tier] = {"success_rate": succ, "crash_rate": crash}

        with open(self.filename, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)

        total = sum(len(v) for v in by_tier.values())
        hits = sum(1 for v in by_tier.values() for r in v if r.success)
        self.last_overall_success = round(hits / total, 4)