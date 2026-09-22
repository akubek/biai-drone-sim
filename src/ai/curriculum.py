import json
from pathlib import Path

from neat.reporting import BaseReporter

from src.core.environment import TIERS


class CurriculumController(BaseReporter):
    """Awans i degradacja tieru na podstawie zamrozonego holdoutu."""

    def __init__(self, state, holdout, rules_path="conf/curriculum.json", baselines_path="data/baselines.json"):
        self.state = state
        self.holdout = holdout
        r = json.loads(Path(rules_path).read_text(encoding="utf-8"))
        self.promote = r["promote"]
        self.demote = r["demote"]
        self.min_dwell = r["min_dwell_gens"]
        self.cooldown = r["cooldown_after_demote_gens"]
        self.generation = 0
        self._tier_since = 0
        self._ok = 0
        self._bad = 0
        self._cooldown_until = -1
        self.baselines = json.loads(
            Path(baselines_path).read_text(encoding="utf-8"))["expert"]

    def start_generation(self, generation: int) -> None:
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome) -> None:
        if not self.state.curriculum_enabled:
            return
        tier = self.state.current_tier
        s = self.holdout.last_by_tier.get(tier)
        if s is None:
            return

        exp = self.baselines[str(tier)]
        need_success = max(self.promote["success_floor"], exp["success"] * self.promote["expert_fraction"])
        allow_crash = max(self.promote["crash_floor"], exp["crash"])
        bad_below = exp["success"] * self.demote["expert_fraction"]

        self._ok = (self._ok + 1
                    if s["success_rate"] >= need_success
                    and s["crash_rate"] <= allow_crash
                    else 0)
        self._bad = self._bad + 1 if s["success_rate"] < bad_below else 0

        if self.generation - self._tier_since < self.min_dwell:
            return

        if (self._ok >= self.promote["consecutive"] and tier < max(TIERS)
                and self.generation >= self._cooldown_until):
            self._switch(tier + 1, species, "AWANS")
        elif self._bad >= self.demote["consecutive"] and tier > min(TIERS):
            self._switch(tier - 1, species, "DEGRADACJA")
            self._cooldown_until = self.generation + self.cooldown

    def _switch(self, new_tier: int, species, label: str) -> None:
        print(f"[curriculum] gen {self.generation}: {label} "
              f"{self.state.current_tier} -> {new_tier}")

        if new_tier < self.state.current_tier:
            self.state.force_scenario_reset()
        else:
            self.state.force_rotation() # advance counter to force scenario rotation at newest refresh
        self.state.current_tier = new_tier
        self._tier_since = self.generation
        self._ok = self._bad = 0
        # zadanie sie zmienilo, wiec stagnacja liczy sie od zera
        for sp in species.species.values():
            sp.last_improved = self.generation
            sp.fitness_history = []