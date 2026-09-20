import csv
import os
import statistics
import time

from neat.reporting import BaseReporter

from src.ai.state import TrainingState
from src.core.stats import EndReason, EpisodeResult

HEADERS = [
    "run_id", "seed", "arch", "net_type", "training_mode",
    "generation", "evaluations_total", "wall_time_s",
    "stage", "num_obstacles", "expert_weight",
    "best_fitness", "mean_fitness", "median_fitness", "std_fitness",
    "success_rate",
    "crash_rate", "escape_rate", "spinout_rate", "stagnation_rate", "timeout_rate",
    "mean_time_to_target", "mean_energy", "mean_min_dist_ratio",
    "species_count", "best_nodes", "best_conns",
]


def _flatten(metrics: dict) -> list[EpisodeResult]:
    """Obsluguje zarowno 1 epizod na genom, jak i liste epizodow (po #15)."""
    out: list[EpisodeResult] = []
    for value in metrics.values():
        out.extend(value) if isinstance(value, list) else out.append(value)
    return out


class CSVTrainingReporter(BaseReporter):
    """Saves generation metrics to CSV. The source of truth is state.last_metrics."""

    def __init__(self, training_state: TrainingState, folder: str = "logs",
                 filename: str = "evolution_log.csv", run_id: str = ""):
        self.state = training_state
        self.run_id = run_id
        os.makedirs(folder, exist_ok=True)
        self.filename = os.path.join(folder, filename)
        self.evaluations_total = 0
        self._gen_start = time.time()
        self._generation = 0

        if not os.path.exists(self.filename):
            with open(self.filename, mode="w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(HEADERS)

    def start_generation(self, generation: int) -> None:
        # Licznik NEAT-a, nie nasz - unika przesuniecia o 1 (state.generation
        # jest inkrementowany przed post_evaluate).
        self._generation = generation
        self._gen_start = time.time()

    def post_evaluate(self, config, population, species, best_genome) -> None:
        results = _flatten(self.state.last_metrics)
        if not results:
            return

        self.evaluations_total += len(results)
        n = len(results)

        def rate(reason: EndReason) -> float:
            return sum(1 for r in results if r.end_reason is reason) / n

        fitnesses = [r.fitness for r in results]
        successes = [r for r in results if r.success]
        exp = self.state.exp_config if hasattr(self.state, "exp_config") else {}

        row = [
            self.run_id,
            exp.get("seed", ""),
            exp.get("arch", ""),
            exp.get("net_type", ""),
            getattr(self.state, "mode", ""),

            self._generation,
            self.evaluations_total,
            round(time.time() - self._gen_start, 2),

            getattr(self.state, "current_stage", ""),
            getattr(self.state, "num_obstacles", ""),
            round(self.state.current_help_weight, 3),

            round(max(fitnesses), 4),
            round(statistics.fmean(fitnesses), 4),
            round(statistics.median(fitnesses), 4),
            round(statistics.pstdev(fitnesses), 4) if n > 1 else 0.0,

            round(len(successes) / n, 4),

            round(rate(EndReason.CRASH), 4),
            round(rate(EndReason.ESCAPE), 4),
            round(rate(EndReason.SPINOUT), 4),
            round(rate(EndReason.STAGNATION), 4),
            round(rate(EndReason.TIMEOUT), 4),

            round(statistics.fmean([r.time_alive_s for r in successes]), 3) if successes else "",
            round(statistics.fmean([r.energy for r in results]), 4),
            round(statistics.fmean([r.min_dist_ratio for r in results]), 4),

            len(species.species) if species else 0,
            len(best_genome.nodes) if best_genome else 0,
            len(best_genome.connections) if best_genome else 0,
        ]

        with open(self.filename, mode="a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)