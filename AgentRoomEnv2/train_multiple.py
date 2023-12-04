"""This script is to tran multiple train.py in parallel.
Things learned:
1. gamma=0.99 is always worse than gamma=0.65
"""
import datetime
import os
import subprocess
from copy import deepcopy

from tqdm import tqdm

from explicit_memory.utils import write_yaml

train_config = {
    "env_str": "room_env:RoomEnv-v2",
    "max_epsilon": 1.0,
    "min_epsilon": 0.1,
    "epsilon_decay_until": 100 * 16,
    "gamma": 0.65,
    "capacity": {
        "episodic": 16,
        "episodic_agent": 16,
        "semantic": 16,
        "short": 1,
    },
    "nn_params": {
        "hidden_size": 64,
        "num_layers": 2,
        "embedding_dim": 32,
        "v1_params": None,
        "v2_params": {},
        "memory_of_interest": [
            "episodic",
            "semantic",
            "short",
        ],
    },
    "num_iterations": 100 * 16,
    "replay_buffer_size": 1024 * 100,
    "warm_start": 1024 * 100,
    "batch_size": 1024,
    "target_update_interval": 10,
    "pretrain_semantic": False,
    "run_test": True,
    "num_samples_for_results": 10,
    "train_seed": 0 + 5,
    "plotting_interval": 10,
    "device": "cpu",
    "test_seed": 0,
    # "mm_policy": "generalize",
    "qa_policy": "episodic_semantic",
    "explore_policy": "avoid_walls",
    "env_config": {
        "question_prob": 1.0,
        "terminates_at": 99,
        "room_size": "xxs",
    },
    "ddqn": False,
    "dueling_dqn": False,
    "split_reward_training": False,
}

commands = []
num_parallel = 2
reverse = False
os.makedirs("./junks", exist_ok=True)

for pretrain_semantic in [False, True]:
    for test_seed in [0, 1, 2, 3, 4]:
        train_config["pretrain_semantic"] = pretrain_semantic
        train_config["test_seed"] = test_seed
        train_config["train_seed"] = test_seed + 5

        config_file_name = (
            f"./junks/{str(datetime.datetime.now()).replace(' ', '-')}.yaml"
        )

        write_yaml(train_config, config_file_name)

        commands.append(f"python train.py --config {config_file_name}")


print(f"Running {len(commands)} training scripts ...")
if reverse:
    commands.reverse()
commands_original = deepcopy(commands)

commands_batched = [
    [commands[i * num_parallel + j] for j in range(num_parallel)]
    for i in range(len(commands) // num_parallel)
]

if len(commands) % num_parallel != 0:
    commands_batched.append(commands[-(len(commands) % num_parallel) :])

assert commands == [bar for foo in commands_batched for bar in foo]


for commands in tqdm(commands_batched):
    procs = [subprocess.Popen(command, shell=True) for command in commands]
    for p in procs:
        p.communicate()
