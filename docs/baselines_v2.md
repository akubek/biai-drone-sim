# Baselines on the holdout set

Reference points for the agent's performance. **Ceiling** is the hand-coded controller
(`HardcodedBrain`), **floor** is the untrained networks with random weights,
i.e., what is seen in the zero-th generation.

| parameter | value |
|---|---|
| date | 2026-09-23 00:37 |
| commit | 646c59e |
| arch | cascade |
| net_type | feedforward |
| seed | 42 |
| ladder | v2 |
| holdout | data/holdout_v2.json |
| scenarios | 100 |
| genomes (floor) | 20 |

**Overall success:** expert 0.580 | random networks 0.000

## Expert (ceiling)

| level | n | success | crash | escape | dist_ratio | time to target [s] |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 0.800 | 0.200 | 0.000 | 0.149 | 3.21 |
| 2 | 20 | 0.650 | 0.350 | 0.000 | 0.281 | 3.95 |
| 3 | 20 | 0.700 | 0.300 | 0.000 | 0.260 | 4.69 |
| 4 | 20 | 0.450 | 0.550 | 0.000 | 0.400 | 4.61 |
| 5 | 20 | 0.300 | 0.700 | 0.000 | 0.505 | 5.26 |

## Untrained networks (floor)

| level | n | success | crash | escape | dist_ratio | time to target [s] |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 400 | 0.000 | 0.810 | 0.190 | 0.798 | – |
| 2 | 400 | 0.000 | 0.890 | 0.107 | 0.810 | – |
| 3 | 400 | 0.000 | 0.965 | 0.022 | 0.863 | – |
| 4 | 400 | 0.000 | 0.963 | 0.037 | 0.890 | – |
| 5 | 400 | 0.000 | 0.950 | 0.040 | 0.930 | – |
