import neat


class AdaptiveThreshold(neat.reporting.BaseReporter):
    """Adjusts compatibility_threshold toward the target number of species."""

    def __init__(self, config, target=12, step=0.005, lo=0.01, hi=5.0):
        self.cfg = config.species_set_config
        self.target = target
        self.step = step
        self.lo = lo
        self.hi = hi

    def end_generation(self, config, population, species_set):
        n = len(species_set.species)
        t = self.cfg.compatibility_threshold
        if n < self.target:
            t -= self.step
        elif n > self.target:
            t += self.step
        self.cfg.compatibility_threshold = min(max(t, self.lo), self.hi)