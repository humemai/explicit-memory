"""This script is to tran multiple train.py in parallel."""
import matplotlib
import datetime

matplotlib.use("Agg")

import logging

logger = logging.getLogger()
logger.disabled = True

import random

from copy import deepcopy
from tqdm.auto import tqdm
from explicit_memory.utils import write_yaml
import os
import subprocess

config = {
    "question_prob": 1.0,
    "terminates_at": 99,
    "randomize_observations": "objects",
    "room_size": "l",
    "rewards": {"correct": 1, "wrong": 0, "partial": 0},
    "make_everything_static": False,
    "num_total_questions": 1000,
    "question_interval": 20,
    "include_walls_in_observations": True,
}

params = {
    "env_str": "room_env:RoomEnv-v2",
    "max_epsilon": 1.0,
    "min_epsilon": 0.1,
    "epsilon_decay_until": 100 * 200,
    "gamma": 0.99,
    "capacity": {
        "episodic": 16,
        "episodic_agent": 0,
        "semantic": 16,
        "semantic_map": 0,
        "short": 1,
    },
    "nn_params": {
        "architecture": "lstm",
        "hidden_size": 64,
        "num_layers": 2,
        "embedding_dim": 64,
        "make_categorical_embeddings": False,
        "v1_params": None,
        "v2_params": {},
        "memory_of_interest": [
            "episodic",
            "semantic",
        ],
        "fuse_information": "sum",
        "include_positional_encoding": True,
        "max_timesteps": config["terminates_at"] + 1,
        "max_strength": config["terminates_at"] + 1,
    },
    "num_iterations": 100 * 200,
    "replay_buffer_size": 100 * 200,
    "warm_start": 100 * 200 / 10,
    "batch_size": 32,
    "target_update_interval": 10,
    "pretrain_semantic": False,
    "run_test": True,
    "num_samples_for_results": 10,
    "train_seed": 5,
    "plotting_interval": 10,
    "device": "cpu",
    "test_seed": 0,
    # "mm_policy": "RL",
    # "mm_agent_path": "trained-agents/lstm-mm/2023-12-28 18:13:03.001952/agent.pkl",
    "qa_policy": "episodic_semantic",
    "explore_policy": "avoid_walls",
    "env_config": config,
    "ddqn": True,
    "dueling_dqn": True,
    "default_root_dir": "training_results/mm/TRASH",
    "run_handcrafted_baselines": [
        {
            "mm": mm,
            "qa": qa,
            "explore": explore,
            "pretrain_semantic": pretrain_semantic,
        }
        for mm in ["random", "episodic", "semantic"]
        for qa in ["episodic_semantic"]
        for explore in ["random", "avoid_walls"]
        for pretrain_semantic in [False, "exclude_walls"]
    ],
}


commands = []
num_parallel = 1
reverse = False
shuffle = True

os.makedirs("./junks", exist_ok=True)

for test_seed in [0, 1, 2, 3, 4]:
    params["test_seed"] = test_seed
    params["train_seed"] = test_seed + 5

    config_file_name = f"./junks/{str(datetime.datetime.now()).replace(' ', '-')}.yaml"

    write_yaml(params, config_file_name)

    commands.append(f"python train.py --config {config_file_name}")


print(f"Running {len(commands)} training scripts ...")
if reverse:
    commands.reverse()
if shuffle:
    random.shuffle(commands)
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
