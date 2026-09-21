from typing import Any, cast

import neat

from src.ai.state import TrainingState
from src.config.config import *
from src.core.environment import generate_scenarios, generate_start_and_target
from src.core.map_generator import generate_grid_obstacles
from src.core.stats import EpisodeResult


class CurriculumParallelEvaluator(neat.ParallelEvaluator):
    def __init__(self, num_workers, eval_function, training_state: TrainingState, timeout=None):
        super().__init__(num_workers, eval_function, timeout)
        self.state = training_state

    def evaluate(self, genomes, config):
        # 1. Przeliczamy parametry dla nadchodzącej generacji na głównym wątku
        self.state.update_parameters()

        scenarios = generate_scenarios(
            count=self.state.scenarios_per_genome,
            num_obstacles=self.state.num_obstacles
        )
        cast(Any, config).shared_scenarios = scenarios
        
        jobs = [
            self.pool.apply_async(self.eval_function, (genome, config))
            for _, genome in genomes
        ]
        
        metrics: dict[int, EpisodeResult] = {}
        for job, (genome_id, genome) in zip(jobs, genomes):
            fitness, result = job.get(timeout=self.timeout)
            cast(Any, genome).fitness = fitness
            metrics[genome_id] = result

        self.state.last_metrics = metrics
        print(f"metryk: {len(metrics)}, genomow: {len(genomes)}") #TODO temporary
        self.state.generation += 1