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

# history_block_size = 6

params = {
    "env_str": "room_env:RoomEnv-v2",
    "num_iterations": 100 * 200,
    "replay_buffer_size": 100 * 200,
    "warm_start": 100 * 200 / 10,
    "batch_size": 32,
    "target_update_interval": 10,
    "epsilon_decay_until": 100 * 200,
    "max_epsilon": 1.0,
    "min_epsilon": 0.1,
    "gamma": 0.9,
    "history_block_size": None,
    "nn_params": {
        "architecture": "lstm",
        "hidden_size": 128,
        "num_layers": 2,
        "embedding_dim": 128,
        "fuse_information": "sum",
        "include_positional_encoding": True,
        "max_timesteps": 100,
        "max_strength": 100,
    },
    "run_test": True,
    "num_samples_for_results": 10,
    "plotting_interval": 10,
    "train_seed": 5,
    "test_seed": 0,
    "device": "cpu",
    "env_config": {
        "question_prob": 1.0,
        "terminates_at": 99,
        "randomize_observations": "objects",
        "room_size": "l",
        "rewards": {"correct": 1, "wrong": 0, "partial": 0},
        "make_everything_static": False,
        "num_total_questions": 1000,
        "question_interval": 1,
        "include_walls_in_observations": True,
    },
    "ddqn": True,
    "dueling_dqn": True,
    "default_root_dir": None,
    # "default_root_dir": f"./training_results/baselines/dqn_lstm/history_block_size={history_block_size}",
    "run_handcrafted_baselines": True,
}

commands = []
num_parallel = 2
reverse = False
shuffle = False

os.makedirs("./junks", exist_ok=True)

for history_block_size in [1]:
    for test_seed in [0, 1, 2, 3, 4]:
        params["test_seed"] = test_seed
        params["train_seed"] = test_seed + 5
        params["history_block_size"] = history_block_size
        params[
            "default_root_dir"
        ] = f"./training_results/baselines/dqn_lstm/history_block_size={history_block_size}"

        config_file_name = (
            f"./junks/{str(datetime.datetime.now()).replace(' ', '-')}.yaml"
        )

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
