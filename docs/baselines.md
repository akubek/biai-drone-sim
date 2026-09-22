# Baselines on the holdout set

Reference points for the agent's performance. **Ceiling** is the hand-coded controller
(`HardcodedBrain`), **floor** is the untrained networks with random weights,
i.e., what is seen in the zero-th generation.

| parameter | value |
|---|---|
| date | 2026-09-22 22:40 |
| commit | 6e8eab5 |
| arch | cascade |
| net_type | feedforward |
| seed | 42 |
| holdout | data/holdout.json |
| scenarios | 120 |
| genomes (floor) | 20 |

**Overall success:** expert 0.583 | random networks 0.000

## Expert (ceiling)

| level | n | success | crash | escape | dist_ratio | time to target [s] |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 1.000 | 0.000 | 0.000 | 0.046 | 3.20 |
| 2 | 20 | 1.000 | 0.000 | 0.000 | 0.045 | 3.97 |
| 3 | 20 | 0.700 | 0.300 | 0.000 | 0.260 | 4.69 |
| 4 | 20 | 0.400 | 0.600 | 0.000 | 0.489 | 5.30 |
| 5 | 20 | 0.250 | 0.750 | 0.000 | 0.595 | 4.82 |
| 6 | 20 | 0.150 | 0.850 | 0.000 | 0.612 | 4.67 |

## Untrained networks (floor)

| level | n | success | crash | escape | dist_ratio | time to target [s] |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 400 | 0.000 | 0.693 | 0.305 | 0.779 | – |
| 2 | 400 | 0.000 | 0.873 | 0.115 | 0.759 | – |
| 3 | 400 | 0.000 | 0.965 | 0.022 | 0.863 | – |
| 4 | 400 | 0.000 | 0.993 | 0.000 | 0.900 | – |
| 5 | 400 | 0.000 | 0.968 | 0.028 | 0.932 | – |
| 6 | 400 | 0.000 | 0.985 | 0.013 | 0.928 | – |
