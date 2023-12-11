# Scores

## xxs

```python
capacity = {
    "episodic": 2,
    "episodic_agent": 0,
    "semantic": 2,
    "semantic_map": 0,
    "short": 1,
}

config = {
    "question_prob": 1.0,
    "terminates_at": 99,
    "randomize_observations": True,
    "room_size": room_size,
    "rewards": {"correct": 1, "wrong": 0, "partial": 0},
    "make_everything_static": False,
    "num_total_questions": 1000,
    "question_interval": 1,
}
params = {
    "env_str": "room_env:RoomEnv-v2",
    "max_epsilon": 1.0,
    "min_epsilon": 0.1,
    "epsilon_decay_until": 100 * 20,
    "gamma": 0.7,
    "capacity": capacity,
    "nn_params": {
        "hidden_size": 128,
        "num_layers": 2,
        "embedding_dim": 64,
        "v1_params": None,
        "v2_params": {},
        "memory_of_interest": [
            "episodic",
            "semantic",
            "short",
        ],
    },
    "num_iterations": 100 * 20,
    "replay_buffer_size": 32768,
    "warm_start": 4096,
    "batch_size": 1024,
    "target_update_interval": 10,
    "pretrain_semantic": False,
    "run_test": True,
    "num_samples_for_results": 10,
    "train_seed": 5,
    "plotting_interval": 10,
    "device": "cpu",
    "test_seed": 0,
    "qa_policy": "episodic_semantic",
    "explore_policy": "avoid_walls",
    "env_config": config,
    "ddqn": True,
    "dueling_dqn": True,
    "split_reward_training": True,
    "default_root_dir": "./training_results/TRASH",
}
```

The number of room: 1\
The number of static_objects: 1\
The number of independent_objects: 1\
The number of dependent_objects: 1

| memory mgmt | qa                | explore     | pretrain_semantic | mean   | std   |
| ----------- | ----------------- | ----------- | ----------------- | ------ | ----- |
| random      | episodic_semantic | avoid_walls | False             | 404.26 | 25.61 |
| random      | episodic_semantic | random      | False             | 394.86 | 25.93 |
| episodic    | episodic_semantic | avoid_walls | False             | 253.36 | 2.38  |
| episodic    | episodic_semantic | random      | False             | 246.52 | 5.53  |
| semantic    | episodic_semantic | avoid_walls | False             | 156.8  | 13.28 |
| semantic    | episodic_semantic | random      | False             | 157.8  | 13.49 |
| RL          | episodic_semantic | avoid_walls | False             | 886    | 113   |

## xs

```python
capacity = {
    "episodic": 4,
    "episodic_agent": 0,
    "semantic": 4,
    "semantic_map": 0,
    "short": 1,
}

config = {
    "question_prob": 1.0,
    "terminates_at": 99,
    "randomize_observations": True,
    "room_size": room_size,
    "rewards": {"correct": 1, "wrong": 0, "partial": 0},
    "make_everything_static": False,
    "num_total_questions": 1000,
    "question_interval": 1,
}
params = {
    "env_str": "room_env:RoomEnv-v2",
    "max_epsilon": 1.0,
    "min_epsilon": 0.1,
    "epsilon_decay_until": 100 * 20,
    "gamma": 0.7,
    "capacity": capacity,
    "nn_params": {
        "hidden_size": 128,
        "num_layers": 2,
        "embedding_dim": 64,
        "v1_params": None,
        "v2_params": {},
        "memory_of_interest": [
            "episodic",
            "semantic",
            "short",
        ],
    },
    "num_iterations": 100 * 20,
    "replay_buffer_size": 32768,
    "warm_start": 4096,
    "batch_size": 1024,
    "target_update_interval": 10,
    "pretrain_semantic": False,
    "run_test": True,
    "num_samples_for_results": 10,
    "train_seed": 5,
    "plotting_interval": 10,
    "device": "cpu",
    "test_seed": 0,
    "qa_policy": "episodic_semantic",
    "explore_policy": "avoid_walls",
    "env_config": config,
    "ddqn": True,
    "dueling_dqn": True,
    "split_reward_training": True,
    "default_root_dir": "./training_results/TRASH",
}
```

The number of room: 1\
The number of static_objects: 2\
The number of independent_objects: 2\
The number of dependent_objects: 2

| memory mgmt | qa                | explore     | pretrain_semantic | mean   | std   |
| ----------- | ----------------- | ----------- | ----------------- | ------ | ----- |
| random      | episodic_semantic | avoid_walls | False             | 613.86 | 57.64 |
| random      | episodic_semantic | random      | False             | 639.18 | 36.33 |
| episodic    | episodic_semantic | avoid_walls | False             | 358.24 | 8.65  |
| episodic    | episodic_semantic | random      | False             | 357.26 | 9.5   |
| semantic    | episodic_semantic | avoid_walls | False             | 418.26 | 55.01 |
| semantic    | episodic_semantic | random      | False             | 544.06 | 55.83 |
| RL          | episodic_semantic | avoid_walls | False             | 833    | 224   |

## s

```python
capacity = {
    "episodic": 8,
    "episodic_agent": 0,
    "semantic": 8,
    "semantic_map": 0,
    "short": 1,
}

config = {
    "question_prob": 1.0,
    "terminates_at": 99,
    "randomize_observations": True,
    "room_size": room_size,
    "rewards": {"correct": 1, "wrong": 0, "partial": 0},
    "make_everything_static": False,
    "num_total_questions": 1000,
    "question_interval": 1,
}
params = {
    "env_str": "room_env:RoomEnv-v2",
    "max_epsilon": 1.0,
    "min_epsilon": 0.1,
    "epsilon_decay_until": 100 * 20,
    "gamma": 0.7,
    "capacity": capacity,
    "nn_params": {
        "hidden_size": 128,
        "num_layers": 2,
        "embedding_dim": 64,
        "v1_params": None,
        "v2_params": {},
        "memory_of_interest": [
            "episodic",
            "semantic",
            "short",
        ],
    },
    "num_iterations": 100 * 20,
    "replay_buffer_size": 32768,
    "warm_start": 4096,
    "batch_size": 1024,
    "target_update_interval": 10,
    "pretrain_semantic": False,
    "run_test": True,
    "num_samples_for_results": 10,
    "train_seed": 5,
    "plotting_interval": 10,
    "device": "cpu",
    "test_seed": 0,
    "qa_policy": "episodic_semantic",
    "explore_policy": "avoid_walls",
    "env_config": config,
    "ddqn": True,
    "dueling_dqn": True,
    "split_reward_training": True,
    "default_root_dir": "./training_results/TRASH",
}
```

The number of room: 2\
The number of static_objects: 4\
The number of independent_objects: 4\
The number of dependent_objects: 4

| memory mgmt | qa                | explore     | pretrain_semantic | mean   | std  |
| ----------- | ----------------- | ----------- | ----------------- | ------ | ---- |
| random      | episodic_semantic | avoid_walls | False             | 524.76 | 5.72 |
| random      | episodic_semantic | random      | False             | 488.72 | 7.84 |
| episodic    | episodic_semantic | avoid_walls | False             | 358.52 | 4.36 |
| episodic    | episodic_semantic | random      | False             | 350.66 | 4.25 |
| semantic    | episodic_semantic | avoid_walls | False             | 317.6  | 8.24 |
| semantic    | episodic_semantic | random      | False             | 310.5  | 7.06 |
| RL          | episodic_semantic | avoid_walls | False             | 436    | 59   |

## m

```python
capacity = {
    "episodic": 16,
    "episodic_agent": 0,
    "semantic": 16,
    "semantic_map": 0,
    "short": 1,
}

config = {
    "question_prob": 1.0,
    "terminates_at": 99,
    "randomize_observations": True,
    "room_size": room_size,
    "rewards": {"correct": 1, "wrong": 0, "partial": 0},
    "make_everything_static": False,
    "num_total_questions": 1000,
    "question_interval": 1,
}
params = {
    "env_str": "room_env:RoomEnv-v2",
    "max_epsilon": 1.0,
    "min_epsilon": 0.1,
    "epsilon_decay_until": 100 * 20,
    "gamma": 0.7,
    "capacity": capacity,
    "nn_params": {
        "hidden_size": 128,
        "num_layers": 2,
        "embedding_dim": 64,
        "v1_params": None,
        "v2_params": {},
        "memory_of_interest": [
            "episodic",
            "semantic",
            "short",
        ],
    },
    "num_iterations": 100 * 20,
    "replay_buffer_size": 32768,
    "warm_start": 4096,
    "batch_size": 1024,
    "target_update_interval": 10,
    "pretrain_semantic": False,
    "run_test": True,
    "num_samples_for_results": 10,
    "train_seed": 5,
    "plotting_interval": 10,
    "device": "cpu",
    "test_seed": 0,
    "qa_policy": "episodic_semantic",
    "explore_policy": "avoid_walls",
    "env_config": config,
    "ddqn": True,
    "dueling_dqn": True,
    "split_reward_training": True,
    "default_root_dir": "./training_results/TRASH",
}
```

The number of room: 4\
The number of static_objects: 8\
The number of independent_objects: 8\
The number of dependent_objects: 8

| memory mgmt | qa                | explore     | pretrain_semantic | mean   | std   |
| ----------- | ----------------- | ----------- | ----------------- | ------ | ----- |
| random      | episodic_semantic | avoid_walls | False             | 387.4  | 14.13 |
| random      | episodic_semantic | random      | False             | 398.04 | 12.57 |
| episodic    | episodic_semantic | avoid_walls | False             | 316.16 | 5.76  |
| episodic    | episodic_semantic | random      | False             | 292.46 | 10.13 |
| semantic    | episodic_semantic | avoid_walls | False             | 218.96 | 13.28 |
| semantic    | episodic_semantic | random      | False             | 237.64 | 18.43 |
| RL          | episodic_semantic | avoid_walls | False             | 217    | 78    |

## l

```python
capacity = {
    "episodic": 32,
    "episodic_agent": 0,
    "semantic": 32,
    "semantic_map": 0,
    "short": 1,
}

config = {
    "question_prob": 1.0,
    "terminates_at": 99,
    "randomize_observations": True,
    "room_size": room_size,
    "rewards": {"correct": 1, "wrong": 0, "partial": 0},
    "make_everything_static": False,
    "num_total_questions": 1000,
    "question_interval": 1,
}
params = {
    "env_str": "room_env:RoomEnv-v2",
    "max_epsilon": 1.0,
    "min_epsilon": 0.1,
    "epsilon_decay_until": 100 * 20,
    "gamma": 0.7,
    "capacity": capacity,
    "nn_params": {
        "hidden_size": 128,
        "num_layers": 2,
        "embedding_dim": 64,
        "v1_params": None,
        "v2_params": {},
        "memory_of_interest": [
            "episodic",
            "semantic",
            "short",
        ],
    },
    "num_iterations": 100 * 20,
    "replay_buffer_size": 32768,
    "warm_start": 4096,
    "batch_size": 1024,
    "target_update_interval": 10,
    "pretrain_semantic": False,
    "run_test": True,
    "num_samples_for_results": 10,
    "train_seed": 5,
    "plotting_interval": 10,
    "device": "cpu",
    "test_seed": 0,
    "qa_policy": "episodic_semantic",
    "explore_policy": "avoid_walls",
    "env_config": config,
    "ddqn": True,
    "dueling_dqn": True,
    "split_reward_training": True,
    "default_root_dir": "./training_results/TRASH",
}
```

The number of room: 8\
The number of static_objects: 16\
The number of independent_objects: 16\
The number of dependent_objects: 16

| memory mgmt | qa                | explore     | pretrain_semantic | mean   | std   |
| ----------- | ----------------- | ----------- | ----------------- | ------ | ----- |
| random      | episodic_semantic | avoid_walls | False             | 357.62 | 13.75 |
| random      | episodic_semantic | random      | False             | 344.02 | 11.38 |
| episodic    | episodic_semantic | avoid_walls | False             | 261.66 | 6.14  |
| episodic    | episodic_semantic | random      | False             | 218.26 | 7.25  |
| semantic    | episodic_semantic | avoid_walls | False             | 208.96 | 4.16  |
| semantic    | episodic_semantic | random      | False             | 206.36 | 7.57  |
| RL          | episodic_semantic | avoid_walls | False             | 225    | 58    |

## xl

```python
capacity = {
    "episodic": 64,
    "episodic_agent": 0,
    "semantic": 64,
    "semantic_map": 0,
    "short": 1,
}

config = {
    "question_prob": 1.0,
    "terminates_at": 99,
    "randomize_observations": True,
    "room_size": room_size,
    "rewards": {"correct": 1, "wrong": 0, "partial": 0},
    "make_everything_static": False,
    "num_total_questions": 1000,
    "question_interval": 1,
}
params = {
    "env_str": "room_env:RoomEnv-v2",
    "max_epsilon": 1.0,
    "min_epsilon": 0.1,
    "epsilon_decay_until": 100 * 20,
    "gamma": 0.7,
    "capacity": capacity,
    "nn_params": {
        "hidden_size": 128,
        "num_layers": 2,
        "embedding_dim": 64,
        "v1_params": None,
        "v2_params": {},
        "memory_of_interest": [
            "episodic",
            "semantic",
            "short",
        ],
    },
    "num_iterations": 100 * 20,
    "replay_buffer_size": 32768,
    "warm_start": 4096,
    "batch_size": 1024,
    "target_update_interval": 10,
    "pretrain_semantic": False,
    "run_test": True,
    "num_samples_for_results": 10,
    "train_seed": 5,
    "plotting_interval": 10,
    "device": "cpu",
    "test_seed": 0,
    "qa_policy": "episodic_semantic",
    "explore_policy": "avoid_walls",
    "env_config": config,
    "ddqn": True,
    "dueling_dqn": True,
    "split_reward_training": True,
    "default_root_dir": "./training_results/TRASH",
}
```

The number of room: 16\
The number of static_objects: 32\
The number of independent_objects: 32\
The number of dependent_objects: 32

| memory mgmt | qa                | explore     | pretrain_semantic | mean   | std  |
| ----------- | ----------------- | ----------- | ----------------- | ------ | ---- |
| random      | episodic_semantic | avoid_walls | False             | 280.42 | 7.56 |
| random      | episodic_semantic | random      | False             | 255.56 | 10.5 |
| episodic    | episodic_semantic | avoid_walls | False             | 197.84 | 1.56 |
| episodic    | episodic_semantic | random      | False             | 163.32 | 5.43 |
| semantic    | episodic_semantic | avoid_walls | False             | 172.4  | 7.8  |
| semantic    | episodic_semantic | random      | False             | 167.7  | 7.35 |
| RL          | episodic_semantic | avoid_walls | False             | 174    | 48   |
