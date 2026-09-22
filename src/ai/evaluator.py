from typing import Any, cast

import neat

from src.ai.state import TrainingState
from src.config.config import *
from src.core.environment import generate_scenarios
from src.core.stats import EpisodeResult


class CurriculumParallelEvaluator(neat.ParallelEvaluator):
    def __init__(self, num_workers, eval_function, training_state: TrainingState, timeout=None):
        super().__init__(num_workers, eval_function, timeout)
        self.state = training_state

    def evaluate(self, genomes, config):
        # 1. Przeliczamy parametry dla nadchodzącej generacji na głównym wątku
        self.state.update_parameters()

        scenarios = self.state.scenarios_for_generation()
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
        self.state.generation += 1