import itertools
import random

import neat

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     "conf/neat-cascade.txt")
pop = neat.Population(config)

sample = random.sample(list(pop.population.values()), 60)
d = sorted(a.distance(b, config.genome_config)
           for a, b in itertools.combinations(sample, 2))

for q in (0.05, 0.20, 0.30, 0.50, 0.75, 0.95):
    print(f"q{int(q*100):02d}: {d[int(q * len(d))]:.3f}")
print(f"max: {d[-1]:.3f}")