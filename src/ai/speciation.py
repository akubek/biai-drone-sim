import neat


class AdaptiveThreshold(neat.reporting.BaseReporter):
    """Adjusts compatibility_threshold toward the target number of species."""

    def __init__(self, target=12, exponent=0.3, lo=0.15, hi=0.6):
        self.target = target
        self.exponent = exponent
        self.lo = lo
        self.hi = hi
        self.current = None 

    def end_generation(self, config, population, species_set):
        cfg = species_set.species_set_config
        n = len(species_set.species)
        t = cfg.compatibility_threshold
        if n < self.target:
            t *= 0.99          # powoli w dół — to tworzy nowe gatunki
        elif n > self.target * 1.5:
            t *= 1.05          # w górę tylko po to, by zatrzymać przyrost
        cfg.compatibility_threshold = min(max(t, self.lo), self.hi)
        reps = [s.representative for s in species_set.species.values()]
        if len(reps) > 1:
            d = [reps[i].distance(reps[j], config.genome_config)
                for i in range(len(reps)) for j in range(i + 1, len(reps))]
            print(f"gen={population and len(species_set.species)} "
                f"thr={species_set.species_set_config.compatibility_threshold:.3f} "
                f"rep_dist min={min(d):.3f} med={sorted(d)[len(d)//2]:.3f} "
                f"max={max(d):.3f}")