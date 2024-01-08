"""Utility functions."""
import csv
import json
import logging
import os
import pickle
import random
import shutil
from collections import deque
from copy import deepcopy
from glob import glob
from typing import Deque, Optional, Union, Literal

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from IPython.display import clear_output
from tqdm.auto import tqdm

from room_env.envs.room2 import RoomEnv2

from .utils import read_pickle, write_yaml
from .policy import answer_question, encode_observation, explore, manage_memory

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def save_figs_and_memories(
    agent_dir: str = "./trained-agents/lstm-explore/2024-01-06 20:04:03.511403/",
    mm_policy: Literal["random", "neural"] = "random",
    qa_policy: Literal[
        "episodic_semantic", "episodic", "semantic"
    ] = "episodic_semantic",
    explore_policy: Literal["random", "avoid_walls", "neural"] = "avoid_walls",
    test_seed: int = 0,
) -> None:
    """Save figures and memories."""
    agent_path = os.path.join(agent_dir, "agent.pkl")
    save_dir = os.path.join(
        agent_dir, f"mm={mm_policy}_qa={qa_policy}_explore={explore_policy}"
    )

    agent = read_pickle(agent_path)
    agent.dqn.eval()
    explore_policy_model = agent.dqn

    agent.env_config["seed"] = test_seed
    env = RoomEnv2(**agent.env_config)

    score = 0
    agent.init_memory_systems()
    observations, info = env.reset()
    env.render("image", save_fig_dir=save_dir)

    for idx, obs in enumerate(observations["room"]):
        encode_observation(agent.memory_systems, obs)
        memory_file_path = os.path.join(
            save_dir,
            str(env.current_time).zfill(3) + "-" + str(idx).zfill(3) + ".yaml",
        )
        write_yaml(
            agent.memory_systems.return_as_a_dict_list(),
            memory_file_path,
        )
        manage_memory(
            agent.memory_systems,
            mm_policy,
            agent.mm_policy_model,
            split_possessive=False,
        )

    while True:
        actions_qa = [
            answer_question(
                agent.memory_systems,
                qa_policy,
                question,
                split_possessive=False,
            )
            for question in observations["questions"]
        ]
        action_explore = explore(
            agent.memory_systems, explore_policy, explore_policy_model
        )

        action_pair = (actions_qa, action_explore)
        (
            observations,
            reward,
            done,
            truncated,
            info,
        ) = env.step(action_pair)
        env.render("image", save_fig_dir=save_dir)
        score += reward
        done = done or truncated

        if done:
            break

        for idx, obs in enumerate(observations["room"]):
            encode_observation(agent.memory_systems, obs)
            memory_file_path = os.path.join(
                save_dir,
                str(env.current_time).zfill(3) + "-" + str(idx).zfill(3) + ".yaml",
            )
            write_yaml(
                agent.memory_systems.return_as_a_dict_list(),
                memory_file_path,
            )
            manage_memory(
                agent.memory_systems,
                mm_policy,
                agent.mm_policy_model,
                split_possessive=False,
            )
        memory_file_path = os.path.join(
            save_dir,
            str(env.current_time).zfill(3) + "-" + str(idx + 1).zfill(3) + ".yaml",
        )
        write_yaml(
            agent.memory_systems.return_as_a_dict_list(),
            memory_file_path,
        )
    print(score)
    shutil.move(save_dir, save_dir + "_score=" + str(score))
