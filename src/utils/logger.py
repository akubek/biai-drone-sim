import csv
import os
from typing import Any
from neat.reporting import BaseReporter

from src.ai.state import TrainingState

class CSVTrainingReporter(BaseReporter):
    """Oficjalny Reporter NEAT, automatycznie zapisujący statystyki po każdej generacji."""
    
    def __init__(self, training_state: TrainingState, folder: str = "logs", filename: str = "training_logs.csv", min_success_fitness: float = 4000.0):
        self.state = training_state
        self.folder = folder
        os.makedirs(self.folder, exist_ok=True)
        self.filename = os.path.join(self.folder, filename)
        self.min_success_fitness = min_success_fitness
        self.headers = [
            "Generation", 
            "Best_Fitness", 
            "Avg_Fitness", 
            "Success_Rate_Perc", 
            "Species_Count",
            "Expert_Help_Weight"
        ]

        # Tworzenie pliku z nagłówkami, jeśli nie istnieje
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.headers)

    def post_evaluate(self, config, population, species, best_genome):
        pop_size = len(population)
        if pop_size == 0:
            return

        # 1. Podliczanie Fitnessu (population to słownik {id: genom})
        fitnesses = [g.fitness for g in population.values() if g.fitness is not None]
        avg_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 0.0

        # 2. Podliczanie Sukcesów
        success_count = sum(1 for g in population.values() if g.fitness is not None and g.fitness >= self.min_success_fitness)
        success_rate = (success_count / pop_size) * 100.0

        # 3. Gatunki
        species_count = len(species.species) if species else 0

        # Zapis do CSV
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                self.state.generation,
                round(best_genome.fitness, 2) if best_genome and best_genome.fitness else 0.0,
                round(avg_fitness, 2),
                round(success_rate, 2),
                species_count,
                round(self.state.current_help_weight, 2)
            ])