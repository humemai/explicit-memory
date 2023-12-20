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
from agent.dqn import DQNMMAgent
from explicit_memory.utils import write_yaml
import os
import subprocess

config = {
    "question_prob": 1.0,
    "terminates_at": 99,
    "randomize_observations": None,
    "room_size": None,
    "rewards": {"correct": 1, "wrong": -1, "partial": -1},
    "make_everything_static": False,
    "num_total_questions": 1000,
    "question_interval": 1,
    "include_walls_in_observations": True,
}

params = {
    "env_str": "room_env:RoomEnv-v2",
    "max_epsilon": 1.0,
    "min_epsilon": 0.1,
    "epsilon_decay_until": 100 * 100,
    "gamma": None,
    "capacity": None,
    "nn_params": {
        "hidden_size": 64,
        "num_layers": 2,
        "embedding_dim": 64,
        "v1_params": None,
        "v2_params": {},
        "memory_of_interest": [
            "episodic",
            "semantic",
            "short",
        ],
        "fuse_information": "sum",
        "include_positional_encoding": True,
        "max_timesteps": config["terminates_at"] + 1,
        "max_strength": config["terminates_at"] + 1,
    },
    "num_iterations": 100 * 100,
    "replay_buffer_size": 100 * 100,
    "warm_start": 100 * 100 / 10,
    "batch_size": 32,
    "target_update_interval": 10,
    "pretrain_semantic": None,
    "run_test": True,
    "num_samples_for_results": 10,
    "train_seed": None,
    "plotting_interval": 10,
    "device": "cpu",
    "test_seed": None,
    "qa_policy": "episodic_semantic",
    "explore_policy": "avoid_walls",
    "env_config": config,
    "ddqn": True,
    "dueling_dqn": True,
    "split_reward_training": None,
    "default_root_dir": None,
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
num_parallel = 2
reverse = False

os.makedirs("./junks", exist_ok=True)

for test_seed in [0, 1, 2, 3, 4]:
    for gamma in [0.9, 0.99]:
        for room_size in ["m", "l"]:
            for split_reward_training in [False]:
                for pretrain_semantic in [False]:
                    for randomize in ["none", "objects"]:
                        if room_size == "m":
                            capacity = {
                                "episodic": 8,
                                "episodic_agent": 0,
                                "semantic": 8,
                                "semantic_map": 0,
                                "short": 1,
                            }
                        else:
                            capacity = {
                                "episodic": 16,
                                "episodic_agent": 0,
                                "semantic": 16,
                                "semantic_map": 0,
                                "short": 1,
                            }

                        config["room_size"] = room_size
                        config["randomize_observations"] = randomize

                        params["capacity"] = capacity
                        params["split_reward_training"] = split_reward_training
                        params["pretrain_semantic"] = pretrain_semantic
                        params["gamma"] = gamma
                        params["test_seed"] = test_seed
                        params["train_seed"] = test_seed + 5

                        params[
                            "default_root_dir"
                        ] = f"./training_results/room_size={room_size}/split_reward_training={split_reward_training}/pretrain_semantic={pretrain_semantic}/randomize={randomize}/gamma={gamma}/"

                        config_file_name = f"./junks/{str(datetime.datetime.now()).replace(' ', '-')}.yaml"

                        write_yaml(params, config_file_name)

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
