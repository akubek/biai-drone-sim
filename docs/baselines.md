# Baselines on the holdout set

Reference points for the agent's performance. **Ceiling** is the hand-coded controller
(`HardcodedBrain`), **floor** is the untrained networks with random weights,
i.e., what is seen in the zero-th generation.

| parameter | value |
|---|---|
| date | 2026-09-21 20:58 |
| commit | 9cb1e6e |
| arch | cascade |
| net_type | feedforward |
| seed | 42 |
| holdout | data/holdout.json |
| scenarios | 120 |
| genomes (floor) | 20 |

**Overall success:** expert 0.808 | random networks 0.000

## Expert (ceiling)

| level | n | success | crash | escape | dist_ratio | time to target [s] |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 1.000 | 0.000 | 0.000 | 0.050 | 2.76 |
| 2 | 20 | 1.000 | 0.000 | 0.000 | 0.047 | 3.29 |
| 3 | 20 | 0.900 | 0.100 | 0.000 | 0.005 | 5.38 |
| 4 | 20 | 0.850 | 0.150 | 0.000 | 0.073 | 4.73 |
| 5 | 20 | 0.650 | 0.350 | 0.000 | 0.249 | 4.98 |
| 6 | 20 | 0.450 | 0.550 | 0.000 | 0.327 | 4.68 |

## Untrained networks (floor)

| level | n | success | crash | escape | dist_ratio | time to target [s] |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 400 | 0.000 | 0.665 | 0.335 | 0.806 | – |
| 2 | 400 | 0.000 | 0.855 | 0.140 | 0.788 | – |
| 3 | 400 | 0.000 | 0.993 | 0.005 | 0.801 | – |
| 4 | 400 | 0.000 | 0.950 | 0.045 | 0.820 | – |
| 5 | 400 | 0.000 | 0.985 | 0.010 | 0.843 | – |
| 6 | 400 | 0.000 | 0.998 | 0.003 | 0.895 | – |
