import csv
import os
import statistics
import time

from neat.reporting import BaseReporter

from src.ai.state import TrainingState
from src.core.stats import EndReason, EpisodeResult

HEADERS = [
    "run_id", "seed", "arch", "net_type", "training_mode",
    "generation", "evaluations_total", "scenarios_per_genome", "episodes_this_gen",  "wall_time_s",
    "stage", "num_obstacles", "expert_weight",
    "best_fitness", "mean_fitness", "median_fitness", "std_fitness",
    "mean_progress", "mean_discovery", "mean_hover", "mean_success",
    "mean_crash_penalty", "mean_kamikaze_penalty",
    "mean_energy_penalty", "mean_shaping",
    "success_rate", "holdout_success_rate", 
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
        episodes = _flatten(self.state.last_metrics)
        if not episodes:
            print("No episodes to log! Skipping CSV entry.")
            return

        self.evaluations_total += len(episodes)

        fitnesses = [g.fitness for g in population.values() if g.fitness is not None]

        n = len(episodes)

        def rate(reason: EndReason) -> float:
            return sum(1 for r in episodes if r.end_reason is reason) / n
        successes = [r for r in episodes if r.success]

        exp = getattr(self.state, "exp_config", {})

        row = [
            self.run_id,    #run_id
            exp.get("seed", ""), #seed
            exp.get("arch", ""), #arch
            exp.get("net_type", ""), #net_type
            getattr(self.state, "mode", ""), #training_mode

            self._generation, #generation
            self.evaluations_total, #evaluations_total
            self.state.scenarios_per_genome, #scenarios_per_genome
            len(episodes),    #episodes_this_gen
            round(time.time() - self._gen_start, 2), #wall_time_s

            getattr(self.state, "current_stage", ""), #stage
            getattr(self.state, "num_obstacles", ""), #num_obstacles
            round(self.state.current_help_weight, 3), #expert_weight

            round(max(fitnesses), 4), #best_fitness
            round(statistics.fmean(fitnesses), 4), #mean_fitness
            round(statistics.median(fitnesses), 4), #median_fitness
            round(statistics.pstdev(fitnesses), 4) if len(fitnesses) > 1 else 0.0, #std_fitness

            round(statistics.fmean(r.components.progress for r in episodes), 3), #mean_progress
            round(statistics.fmean(r.components.discovery for r in episodes), 3), #mean_discovery
            round(statistics.fmean(r.components.hover for r in episodes), 3), #mean_hover
            round(statistics.fmean(r.components.success for r in episodes), 3), #mean_success
            round(statistics.fmean(r.components.crash_penalty for r in episodes), 3), #mean_crash_penalty
            round(statistics.fmean(r.components.kamikaze_penalty for r in episodes), 3), #mean_kamikaze_penalty
            round(statistics.fmean(r.components.energy_penalty for r in episodes), 3), #mean_energy_penalty
            round(statistics.fmean(r.components.shaping for r in episodes), 3), #mean_shaping

            round(len(successes) / n, 4), #success_rate
            getattr(getattr(self, "holdout", None), "last_overall_success", ""), #holdout_success_rate

            round(rate(EndReason.CRASH), 4), #crash_rate
            round(rate(EndReason.ESCAPE), 4), #escape_rate
            round(rate(EndReason.SPINOUT), 4), #spinout_rate
            round(rate(EndReason.STAGNATION), 4), #stagnation_rate
            round(rate(EndReason.TIMEOUT), 4), #timeout_rate

            round(statistics.fmean([r.time_alive_s for r in successes]), 3) if successes else "", #mean_time_to_target
            round(statistics.fmean([r.energy for r in episodes]), 4), #mean_energy
            round(statistics.fmean([r.min_dist_ratio for r in episodes]), 4), #mean_min_dist_ratio

            len(species.species) if species else 0, #species_count
            len(best_genome.nodes) if best_genome else 0, #best_nodes
            len(best_genome.connections) if best_genome else 0, #best_conns
        ]

        assert len(row) == len(HEADERS), \
            f"row has {len(row)} fields, HEADERS {len(HEADERS)} - columns are misaligned"

        with open(self.filename, mode="a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)