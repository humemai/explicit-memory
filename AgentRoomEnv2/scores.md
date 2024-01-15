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

<!-- ## l2

The number of room: 16\
The number of static_objects: 8\
The number of independent_objects: 8\
The number of dependent_objects: 8

```python
{
    "episodic": 16,
    "episodic_agent": 0,
    "semantic": 16,
    "semantic_map": 0,
    "short": 1,
}
```

| memory mgmt | qa                | explore     | pretrain_semantic | test_mean | test_std |
| ----------- | ----------------- | ----------- | ----------------- | --------- | -------- |
| random      | episodic_semantic | random      | False             | 124.47    | 3.62     |
| random      | episodic_semantic | avoid_walls | False             | 158.67    | 3.39     |
| episodic    | episodic_semantic | random      | False             | 92.24     | 1.26     |
| episodic    | episodic_semantic | avoid_walls | False             | 135.48    | 1.63     |
| semantic    | episodic_semantic | random      | False             | 47.95     | 3.32     |
| semantic    | episodic_semantic | avoid_walls | False             | 48.73     | 2.87     |
| RL          | episodic_semantic | avoid_walls | False             | 214.05    | 31.87    | -->

<!-- ## l, 8

The number of room: 16\
The number of static_objects: 4\
The number of independent_objects: 4\
The number of dependent_objects: 4

| memory mgmt | qa                | explore     | pretrain_semantic | test_mean | test_std |
| ----------- | ----------------- | ----------- | ----------------- | --------- | -------- |
| random      | episodic_semantic | random      | False             | 139.82    | 8.19     |
| random      | episodic_semantic | avoid_walls | False             | 196.54    | 5.30     |
| episodic    | episodic_semantic | random      | False             | 75.01     | 2.43     |
| episodic    | episodic_semantic | avoid_walls | False             | 86.02     | 1.38     |
| semantic    | episodic_semantic | random      | False             | 78.05     | 1.35     |
| semantic    | episodic_semantic | avoid_walls | False             | 121.21    | 6.37     | -->

<!-- ## l2, 8

The number of room: 16\
The number of static_objects: 8\
The number of independent_objects: 8\
The number of dependent_objects: 8

| memory mgmt | qa                | explore     | pretrain_semantic | test_mean | test_std |
| ----------- | ----------------- | ----------- | ----------------- | --------- | -------- |
| random      | episodic_semantic | random      | False             | 72.32     | 1.82     |
| random      | episodic_semantic | avoid_walls | False             | 94.47     | 1.75     |
| episodic    | episodic_semantic | random      | False             | 61.51     | 0.46     |
| episodic    | episodic_semantic | avoid_walls | False             | 74.14     | 0.76     |
| semantic    | episodic_semantic | random      | False             | 14.79     | 1.48     |
| semantic    | episodic_semantic | avoid_walls | False             | 9.81      | 0.87     | -->

<!--

## l2

The number of room: 16\
The number of static_objects: 8\
The number of independent_objects: 8\
The number of dependent_objects: 8

```python
capacity = {
    "episodic": 16,
    "episodic_agent": 0,
    "semantic": 16,
    "semantic_map": 0,
    "short": 1,
}
```

| memory mgmt | qa                | explore     | pretrain_semantic | test_mean | test_std |
| ----------- | ----------------- | ----------- | ----------------- | --------- | -------- |
| random      | episodic_semantic | random      | False             | 124.60    | 3.89     |
| random      | episodic_semantic | avoid_walls | False             | 160.87    | 1.57     |
| episodic    | episodic_semantic | random      | False             | 92.24     | 1.26     |
| episodic    | episodic_semantic | avoid_walls | False             | 135.48    | 1.63     |
| semantic    | episodic_semantic | random      | False             | 49.60     | 1.59     |
| semantic    | episodic_semantic | avoid_walls | False             | 67.17     | 3.06     |

## l2

The number of room: 16\
The number of static_objects: 8\
The number of independent_objects: 8\
The number of dependent_objects: 8

```python
capacity = {
    "episodic": 32,
    "episodic_agent": 0,
    "semantic": 32,
    "semantic_map": 0,
    "short": 1,
}
```

| memory mgmt | qa                | explore     | pretrain_semantic | test_mean | test_std |
| ----------- | ----------------- | ----------- | ----------------- | --------- | -------- |
| random      | episodic_semantic | random      | False             | 191.64    | 6.88     |
| random      | episodic_semantic | avoid_walls | False             | 240.37    | 5.63     |
| episodic    | episodic_semantic | random      | False             | 130.94    | 2.22     |
| episodic    | episodic_semantic | avoid_walls | False             | 198.20    | 5.10     |
| semantic    | episodic_semantic | random      | False             | 105.97    | 2.95     |
| semantic    | episodic_semantic | avoid_walls | False             | 100.33    | 3.25     |

## m

The number of room: 8\
The number of static_objects: 2\
The number of independent_objects: 2\
The number of dependent_objects: 2

| memory mgmt | qa                | explore     | mean   | std  |
| ----------- | ----------------- | ----------- | ------ | ---- |
| random      | episodic_semantic | random      | 98.65  | 6.39 |
| random      | episodic_semantic | avoid_walls | 174.34 | 6.00 |
| episodic    | episodic_semantic | random      | 99.08  | 4.79 |
| episodic    | episodic_semantic | avoid_walls | 161.62 | 6.33 |
| semantic    | episodic_semantic | random      | 14.17  | 1.50 |
| semantic    | episodic_semantic | avoid_walls | 3.72   | 0.79 |

## xl

The number of room: 32\
The number of static_objects: 8\
The number of independent_objects: 8\
The number of dependent_objects: 8

| memory mgmt | qa                | explore     | test_mean | test_std |
| ----------- | ----------------- | ----------- | --------- | -------- |
| random      | episodic_semantic | random      | 66.85     | 2.89     |
| random      | episodic_semantic | avoid_walls | 81.70     | 2.02     |
| episodic    | episodic_semantic | random      | 34.60     | 1.42     |
| episodic    | episodic_semantic | avoid_walls | 39.86     | 0.63     |
| semantic    | episodic_semantic | random      | 22.28     | 2.54     |
| semantic    | episodic_semantic | avoid_walls | 75.00     | 0.77     | -->
