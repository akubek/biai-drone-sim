from dataclasses import dataclass
from enum import Enum


class EndReason(str, Enum):
    """Reason for the end of the episode. Exclusive - exactly one per flight."""
    SUCCESS = "success"
    CRASH = "crash"
    ESCAPE = "escape"
    SPINOUT = "spinout"
    STAGNATION = "stagnation"
    TIMEOUT = "timeout"


@dataclass
class EpisodeResult:
    """Summary of a single flight of a single drone. Filled at the end of the episode."""
    end_reason: EndReason
    fitness: float
    time_alive_s: float
    min_dist_ratio: float
    max_hover_time_s: float
    energy: float
    touched_target: bool

    @property
    def success(self) -> bool:
        return self.end_reason is EndReason.SUCCESS

    @classmethod
    def from_stats(cls, stats: "EvolutionStats", fitness: float,
                    reason: EndReason, max_episode_time_s: float) -> "EpisodeResult":
        # Normalised energy: 1.0 = both engines at full power for the entire episode.
        max_energy = 2.0 * max_episode_time_s
        return cls(
            end_reason=reason,
            fitness=fitness,
            time_alive_s=stats.total_time_alive,
            min_dist_ratio=stats.min_dist_m / stats.initial_dist_m if stats.initial_dist_m > 0 else 0.0,
            max_hover_time_s=stats.max_hover_time_achieved,
            energy=stats.energy_raw / max_energy if max_energy > 0 else 0.0,
            touched_target=stats.has_touched_target,
        )

@dataclass
class EvolutionStats:
    """Stores the state and progress of the drone in the environment."""
    initial_dist_m: float = 0.0
    min_dist_m: float = 0.0
    max_allowed_escape_dist_m: float = 0.0
    hover_time: float = 0.0
    last_stagnation_dist_m: float = 0.0
    time_without_progress: float = 0.0
    total_time_alive: float = 0.0
    energy_raw: float = 0.0
    max_hover_time_achieved: float = 0.0
    has_touched_target: bool = False

    spinout_time: float = 0.0