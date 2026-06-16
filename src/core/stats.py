from dataclasses import dataclass

@dataclass
class EvolutionStats:
    """Przechowuje stan i postępy drona w środowisku."""
    initial_dist_m: float = 0.0
    min_dist_m: float = 0.0
    max_allowed_escape_dist_m: float = 0.0
    hover_time: float = 0.0
    last_stagnation_dist_m: float = 0.0
    time_without_progress: float = 0.0
    total_time_alive: float = 0.0
    max_hover_time_achieved: float = 0.0
    has_touched_target: bool = False
    accumulated_rotation: float = 0.0

    idle_time: float = 0.0
    spinout_time: float = 0.0