"""Calculates fitness from episode facts.

Separates responsibility: the simulation loop collects facts into EvolutionStats,
this function converts them into points - once, at the end of the episode.

Rewards for progress and hover are integrals over the trajectory so they are accumulated
in stats.progress_raw / stats.hover_raw during the flight.
"""

from src.config.evolution import *
from src.config.physics import SAFE_CRASH_SPEED_M_S
from src.config.rewards import *
from src.core.stats import EndReason, EvolutionStats, FitnessComponents


def compute_fitness(stats: EvolutionStats, reason: EndReason) -> FitnessComponents:
    """Computes the fitness components from the episode statistics."""

    # Jaki ulamek poczatkowego dystansu udalo sie zamknac.
    if stats.initial_dist_m > 0:
        progress_ratio = 1.0 - (stats.min_dist_m / stats.initial_dist_m)
    else:
        progress_ratio = 0.0
    progress_ratio = max(0.0, min(1.0, progress_ratio))

    hover_ratio = min(stats.max_hover_credit_s  / HOVER_REQUIRED_SEC, 1.0)

    components = FitnessComponents(
        progress=FIT_W_PROGRESS * progress_ratio,
        discovery=FIT_W_DISCOVERY if stats.has_touched_target else 0.0,
        hover=FIT_W_HOVER * hover_ratio,
        success=FIT_W_SUCCESS if reason is EndReason.SUCCESS else 0.0,
    )
    components.energy_penalty = -FIT_W_ENERGY * stats.mean_throttle

    if reason is EndReason.CRASH:
        earned = components.progress + components.discovery
        components.crash_penalty = -FIT_CRASH_FORFEIT * earned
        if stats.crash_speed > SAFE_CRASH_SPEED_M_S:
            components.kamikaze_penalty = -(FIT_KAMIKAZE_FORFEIT
                                            - FIT_CRASH_FORFEIT) * earned

    return components

def hover_credit(speed: float, ang_speed: float) -> float:
    return _ramp(speed, HOVER_MAX_SPEED_M_S, V_REF) \
         * _ramp(abs(ang_speed), HOVER_MAX_ANGULAR_VEL, W_REF)

def _ramp(x: float, lo: float, hi: float) -> float:
    if x <= lo:  return 1.0
    if x >= hi:  return 0.0
    return (hi - x) / (hi - lo)