# Scores

## l

```python
capacity = {
    "episodic": 16,
    "episodic_agent": 0,
    "semantic": 16,
    "semantic_map": 0,
    "short": 1,
}
```

The number of room: 16\
The number of static_objects: 4\
The number of independent_objects: 4\
The number of dependent_objects: 4

### memory-based LSTM agent

| memory mgmt | qa                | explore     | val_mean | val_std | test_mean | test_std | num_params |
| ----------- | ----------------- | ----------- | -------- | ------- | --------- | -------- | ---------- |
| random      | episodic_semantic | random      |          |         | 211.26    | 9.17     |            |
| random      | episodic_semantic | avoid_walls |          |         | 258.51    | 6.79     |            |
| episodic    | episodic_semantic | random      |          |         | 105.78    | 3.20     |            |
| episodic    | episodic_semantic | avoid_walls |          |         | 148.32    | 1.62     |            |
| semantic    | episodic_semantic | random      |          |         | 117.68    | 2.57     |            |
| semantic    | episodic_semantic | avoid_walls |          |         | 147.16    | 5.71     |            |
| RL          | episodic_semantic | avoid_walls | 361.42   | 30.75   | 341.7     | 34.94    | 210,564    |
| neural      | episodic_semantic | RL          | 450.52   | 30.31   | 442.7     | 50.45    | 144,134    |

### history-based LSTM agent

average observations per room (block): 5.71

| history_block_size | avg_obs_size | explore     | val_mean | val_std | test_mean | test_std | num_params |
| ------------------ | ------------ | ----------- | -------- | ------- | --------- | -------- | ---------- |
| 1                  | 5.71         | random      |          |         | 55.8      | 26.46    |            |
| 1                  | 5.71         | avoid_walls |          |         | 62.0      | 20.24    |            |
| 1                  | 5.71         | RL          | 261.2    | 5.08    | 214.94    | 19.94    | 302,598    |
| 6                  | 34.26        | random      |          |         | 143.4     | 43.34    |            |
| 6                  | 34.26        | avoid_walls |          |         | 238.8     | 52.76    |            |
| 6                  | 34.26        | RL          | 282.14   | 59.24   | 309.62    | 63.30    | 302,598    |
| 12                 | 68.52        | random      |          |         | 207.6     | 54.04    |            |
| 12                 | 68.52        | avoid_walls |          |         | 239.8     | 106.87   |            |
| 12                 | 68.52        | RL          | 393.82   | 109.58  | 377.94    | 113.98   | 302,598    |
| 24                 | 137.04       | random      |          |         | 272.8     | 74.70    |            |
| 24                 | 137.04       | avoid_walls |          |         | 385.8     | 133.21   |            |
| 24                 | 137.04       | RL          | 437.03   | 90.85   | 396.43    | 87.84    | 302,598    |
| 48                 | 274.08       | random      |          |         | 361.0     | 103.62   |            |
| 48                 | 274.08       | avoid_walls |          |         | 474.8     | 59.63    |            |
| 48                 | 274.08       | RL          | 429.77   | 85.48   | 379.97    | 95.76    | 302,598    |
| 100                | 571          | random      |          |         | 417.6     | 143.32   |            |
| 100                | 571          | avoid_walls |          |         | 498.6     | 61.25    |            |
| 100                | 571          | RL          | 279.64   | 161.38  | 246.26    | 160.85   | 302,598    |
