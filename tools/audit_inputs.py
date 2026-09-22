# tools/audit_inputs.py
import neat
import numpy as np

from src.ai import neat_eval
from src.ai.expert import HardcodedBrain
from src.core.drone import Drone
from src.core.environment import generate_scenarios

LABELS = ([f"sensor{i}" for i in range(8)] +
          ["vx", "vy", "ang_vel", "sin_ang", "cos_ang",
           "dist", "sin_targ", "cos_targ"])

samples: list[list[float]] = []
_orig = Drone.get_inputs


def _patched(self, *args, **kwargs):
    out = _orig(self, *args, **kwargs)
    samples.append(list(out))
    return out


Drone.get_inputs = _patched

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     "conf/neat-cascade.txt")
pop = neat.Population(config)
expert = HardcodedBrain()                 # dopasuj, jesli bierze argumenty
scenarios = generate_scenarios(3, tier=1)
config.net_type = "feedforward"
config.use_cascade = True

for genome in list(pop.population.values())[:30]:
    for sc in scenarios:
        neat_eval._run_episode(genome, config, sc, expert, 0.0)

if not samples:
    raise SystemExit("0 probek - patch nie dotarl do petli epizodu")

a = np.array(samples)
print(f"{len(samples)} probek, {a.shape[1]} wejsc\n")
for i, name in enumerate(LABELS[:a.shape[1]]):
    c = a[:, i]
    flag = ""
    if abs(c).max() > 5.0:
        flag = "  <-- NASYCENIE"
    elif c.std() < 0.01:
        flag = "  <-- WEJSCIE STALE"
    print(f"{name:10s} min={c.min():7.3f} max={c.max():7.3f} "
          f"mean={c.mean():7.3f} std={c.std():6.3f} "
          f"p90={np.percentile(c, 90):7.3f}{flag}")