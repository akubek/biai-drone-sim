import csv
import json
import os
import pickle
import statistics
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
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
                 config: Any, folder: str, every: int = 10, top_k: int = 3):
        self.scenarios = scenarios
        self.run_episode = run_episode_fn
        self.config = config
        self.every = every
        self.generation = 0
        self.last_overall_success = ""      # read by CSVTrainingReporter
        self.filename = os.path.join(folder, "holdout_log.csv")
        self.last_by_tier: dict[int, dict] = {} 
        self.top_k = top_k
        self._best_score = 0.0
        self.run_dir = folder
        with open(self.filename, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HEADERS)

    def start_generation(self, generation: int) -> None:
        self.generation = generation
        self.last_overall_success = ""
        self.last_by_tier = {}

    def post_evaluate(self, config, population, species, best_genome) -> None:
        if best_genome is None or self.generation % self.every != 0:
            return

        top = sorted((g for g in population.values() if g.fitness is not None),
                 key=lambda g: g.fitness, reverse=True)[:self.top_k]

        if not top:
            return

        best_by_tier: dict[int, list[EpisodeResult]] = {}
        best_genome_sel = None  
        best_hits = -1
        for genome in top:
            by_tier: dict[int, list[EpisodeResult]] = defaultdict(list)
            for scenario in self.scenarios:
                by_tier[scenario.tier].append(
                    self.run_episode(genome, self.config, scenario, None, 0.0))
            hits = sum(1 for v in by_tier.values() for r in v if r.success)
            if hits > best_hits:
                best_hits, best_by_tier = hits, by_tier
                best_genome_sel = genome  
                


        rows = []
        for tier in sorted(best_by_tier):
            results = best_by_tier[tier]
            n = len(results)
            succ = round(sum(1 for r in results if r.success) / n, 4)
            crash = round(sum(1 for r in results
                            if r.end_reason is EndReason.CRASH) / n, 4)
            dist = round(statistics.fmean(r.min_dist_ratio for r in results), 4)

            rows.append([self.generation, tier, n, succ, dist, crash])
            self.last_by_tier[tier] = {"success_rate": succ, "crash_rate": crash}

        with open(self.filename, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)

        total = sum(len(v) for v in best_by_tier.values())
        hits = sum(1 for v in best_by_tier.values() for r in v if r.success)
        self.last_overall_success = round(hits / total, 4)

        if self.last_overall_success > self._best_score:
            self._best_score = self.last_overall_success
            stem = f"best_g{self.generation:03d}_score{self._best_score:.3f}"
            run_dir = Path(self.run_dir)
            with open(run_dir / f"{stem}.pkl", "wb") as fh:
                pickle.dump(best_genome_sel, fh)
            with open(run_dir / "best_holdout_meta.json", "w", encoding="utf-8") as fh:
                json.dump({"generation": self.generation,
                           "score": self._best_score,
                           "per_tier": self.last_by_tier}, fh, indent=2)