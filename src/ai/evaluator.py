import neat

from src.ai.state import TrainingState
from src.config.config import *
from src.core.environment import generate_start_and_target
from src.core.map_generator import generate_grid_obstacles

class CurriculumParallelEvaluator(neat.ParallelEvaluator):
    def __init__(self, num_workers, eval_function, training_state: TrainingState, timeout=None):
        super().__init__(num_workers, eval_function, timeout)
        self.state = training_state

    def evaluate(self, genomes, config):
        # 1. Przeliczamy parametry dla nadchodzącej generacji na głównym wątku
        self.state.update_parameters()

        # Losowanie mapy
        start_px, target_px = generate_start_and_target(
            SCREEN_WIDTH, SCREEN_HEIGHT, MAP_MARGIN_PX, MIN_SPAWN_DIST_M
        )
    
        obstacles = generate_grid_obstacles(SCREEN_WIDTH, SCREEN_HEIGHT, start_px, target_px, GRID_SIZE_M, self.state.num_obstacles, SAFE_ZONE_CELLS, PPM)
        
        # 2. WSTRZYKIWANIE CONFIGU: Dynamicznie doklejamy pola do obiektu NEAT
        # Dzięki temu zostaną one bezpiecznie skopiowane do każdego procesu roboczego!
        config.current_help_weight = self.state.current_help_weight
        config.current_stage = self.state.current_stage

        config.shared_start_px = start_px
        config.shared_target_px = target_px

        config.shared_obstacles_data = [
            (rect.x, rect.y, rect.width, rect.height) for rect in obstacles
        ]
        
        # 3. Odpalamy oryginalne, wieloprocesowe ocenianie genomów
        super().evaluate(genomes, config)
        
        # 4. Zwiększamy licznik generacji po zakończeniu obliczeń
        self.state.generation += 1