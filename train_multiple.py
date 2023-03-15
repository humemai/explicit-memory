"""This script is to tran multiple train.py in parallel.
Things learned:
1. gamma=0.99 is always worse than gamma=0.65
"""
import datetime
import os
import subprocess
from copy import deepcopy

from tqdm import tqdm

from utils import write_yaml

train_config = {
    "allow_random_human": False,
    "allow_random_question": False,
    "pretrain_semantic": False,
    "varying_rewards": False,
    "seed": 0,
    "num_eval_iter": 10,
    "max_epochs": 16,
    "batch_size": 1024,
    "epoch_length": 131072,
    "replay_size": 131072,
    "warm_start_size": 131072,
    "eps_end": 0,
    "eps_last_step": 2048,
    "eps_start": 1.0,
    "gamma": 0.65,
    "lr": 0.001,
    "sync_rate": 10,
    "loss_function": "huber",
    "optimizer": "adam",
    "des_size": "l",
    "capacity": {"episodic": 16, "semantic": 16, "short": 1},
    "question_prob": 0.5,
    "observation_params": "perfect",
    "nn_params": {
        "architecture": "lstm",
        "embedding_dim": 32,
        "hidden_size": 64,
        "include_human": "sum",
        "memory_systems": ["episodic", "semantic", "short"],
        "num_layers": 2,
        "human_embedding_on_object_location": False,
    },
    "log_every_n_steps": 1,
    "early_stopping_patience": 1000,
    "precision": 32,
    "accelerator": "cpu",
}

commands = []
num_parallel = 4
reverse = False
os.makedirs("./junks", exist_ok=True)

for capacity in [2, 4, 8, 16, 32, 64]:
    for pretrain_semantic in [False, True]:
        for seed in [0, 1, 2, 3, 4]:
            train_config["question_prob"] = 0.5
            train_config["capacity"] = {
                "episodic": capacity // 2,
                "semantic": capacity // 2,
                "short": 1,
            }
            train_config["pretrain_semantic"] = pretrain_semantic
            train_config["seed"] = seed

            config_file_name = (
                f"./junks/{str(datetime.datetime.now()).replace(' ', '-')}.yaml"
            )

            write_yaml(train_config, config_file_name)

            commands.append(f"python train.py --config {config_file_name}")

for capacity in [2, 4, 8, 16, 32, 64]:
    for pretrain_semantic in [False, True]:
        for seed in [0, 1, 2, 3, 4]:
            train_config["question_prob"] = 1.0
            train_config["capacity"] = {
                "episodic": capacity // 2,
                "semantic": capacity // 2,
                "short": 1,
            }
            train_config["pretrain_semantic"] = pretrain_semantic
            train_config["seed"] = seed

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
