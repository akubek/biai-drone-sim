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
class FitnessComponents:
    """Breakdown of fitness into components - for diagnostics and logging."""
    progress: float = 0.0
    discovery: float = 0.0
    hover: float = 0.0
    success: float = 0.0
    crash_penalty: float = 0.0        # ujemny
    kamikaze_penalty: float = 0.0     # ujemny
    escape_penalty: float = 0.0       # ujemny
    energy_penalty: float = 0.0       # ujemny
    spinout_penalty: float = 0.0       # ujemny
    shaping: float = 0.0              # wchodzi w #29
    braking: float = 0.0

    @property
    def total(self) -> float:
        return (self.progress + self.discovery + self.hover + self.success
                + self.crash_penalty + self.kamikaze_penalty
                + self.escape_penalty + self.energy_penalty + self.spinout_penalty 
                + self.shaping + self.braking)

@dataclass
class EpisodeResult:
    """Summary of a single flight of a single drone. Filled at the end of the episode."""
    end_reason: EndReason
    fitness: float
    components: FitnessComponents
    time_alive_s: float
    min_dist_ratio: float
    max_hover_time_s: float
    energy: float
    touched_target: bool
    mean_throttle: float
    mean_angular_speed: float
    speed_at_min_dist: float
    ang_speed_at_min_dist: float
    max_lin_credit_s: float
    max_ang_credit_s: float

    @property
    def success(self) -> bool:
        return self.end_reason is EndReason.SUCCESS

    @classmethod
    def from_stats(cls, stats: "EvolutionStats", components : FitnessComponents,
                    reason: EndReason, max_episode_time_s: float) -> "EpisodeResult":
        # Normalised energy: 1.0 = both engines at full power for the entire episode.
        max_energy = 2.0 * max_episode_time_s
        return cls(
            end_reason=reason,
            fitness=components.total,
            components=components,
            time_alive_s=stats.total_time_alive,
            min_dist_ratio=stats.min_dist_m / stats.initial_dist_m if stats.initial_dist_m > 0 else 0.0,
            max_hover_time_s=stats.max_hover_time_achieved,
            energy=stats.energy_raw / max_energy if max_energy > 0 else 0.0,
            touched_target=stats.has_touched_target,
            mean_throttle=stats.mean_throttle,
            mean_angular_speed=stats.mean_angular_speed,
            speed_at_min_dist=stats.speed_at_min_dist,
            ang_speed_at_min_dist=stats.ang_speed_at_min_dist,
            max_lin_credit_s=stats.max_lin_credit_s,
            max_ang_credit_s=stats.max_ang_credit_s,
        )

@dataclass
class EvolutionStats:
    """Stores the state and progress of the drone in the environment."""
    initial_dist_m: float = 0.0
    min_dist_m: float = 0.0
    max_allowed_escape_dist_m: float = 0.0
    hover_time_s: float = 0.0
    last_stagnation_dist_m: float = 0.0
    time_without_progress: float = 0.0
    total_time_alive: float = 0.0
    energy_raw: float = 0.0
    max_hover_time_achieved: float = 0.0
    accumulated_rotation: float = 0.0
    has_touched_target: bool = False
    crash_speed: float = 0.0
    speed_at_min_dist: float = 0.0
    ang_speed_at_min_dist: float = 0.0
    hover_credit_s: float = 0.0
    max_hover_credit_s: float = 0.0
    spinout_time: float = 0.0
    lin_credit_s: float = 0.0
    ang_credit_s: float = 0.0
    max_lin_credit_s: float = 0.0
    max_ang_credit_s: float = 0.0

    def __post_init__(self) -> None:
        if self.min_dist_m <= 0.0:
            self.min_dist_m = self.initial_dist_m

    @property
    def mean_throttle(self) -> float:
        """Mean throttle as a continuous value between 0 and 1, independent of episode length."""
        return (self.energy_raw / (2.0 * self.total_time_alive)
                if self.total_time_alive > 0 else 0.0)
    @property
    def mean_angular_speed(self) -> float:
        """Average |omega| over the entire flight [rad/s]. Captures spinout, slow rotation, and wobbling."""
        return (self.accumulated_rotation / self.total_time_alive
                if self.total_time_alive > 0 else 0.0)

    def observe_distance(
            self, dist_m: float,
            speed_m_s: float,
            ang_speed: float,
        ) -> None:
        if dist_m < self.min_dist_m:
            self.min_dist_m = dist_m
            self.speed_at_min_dist = speed_m_s
            self.ang_speed_at_min_dist = abs(ang_speed)