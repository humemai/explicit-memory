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

| memory mgmt | qa                | explore     | test_mean | test_std |
| ----------- | ----------------- | ----------- | --------- | -------- |
| random      | episodic_semantic | random      | 211.26    | 9.17     |
| random      | episodic_semantic | avoid_walls | 258.51    | 6.79     |
| episodic    | episodic_semantic | random      | 105.78    | 3.20     |
| episodic    | episodic_semantic | avoid_walls | 148.32    | 1.62     |
| semantic    | episodic_semantic | random      | 117.68    | 2.57     |
| semantic    | episodic_semantic | avoid_walls | 147.16    | 5.71     |
| RL          | episodic_semantic | avoid_walls | 341.7     | 34.94    |
| neural      | episodic_semantic | RL          | 442.7     | 50.45    |

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
