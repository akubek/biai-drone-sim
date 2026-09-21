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
    """Fitness computation from episode statistics."""
    components = FitnessComponents(
        progress=stats.progress_raw,
        hover=stats.hover_raw,
        discovery=FIT_DISCOVERY_BONUS if stats.has_touched_target else 0.0,
        success=FIT_HOVER_SUCCESS_REWARD if reason is EndReason.SUCCESS else 0.0,
    )

    if reason is EndReason.CRASH:
        components.crash_penalty = -FIT_CRASH_BASE_PENALTY
        if stats.crash_speed > SAFE_CRASH_SPEED_M_S:
            components.kamikaze_penalty = -FIT_KAMIKAZE_PENALTY

    return components