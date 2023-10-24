"""Handcrafted Agent for the RoomEnv2 environment."""
import datetime
import os
import random
import shutil
from copy import deepcopy
from typing import Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from tqdm.auto import tqdm, trange

from explicit_memory.memory import (
    EpisodicMemory,
    MemorySystems,
    SemanticMemory,
    ShortMemory,
)
from explicit_memory.policy import (
    answer_question,
    encode_observation,
    explore,
    manage_memory,
)
from explicit_memory.utils import write_yaml


class HandcraftedAgent:
    """Handcrafted agent interacting with environment.

    This agent explores the roooms, i.e., KGs. The exploration can be uniform-random,
    or just avoiding walls.

    """

    def __init__(
        self,
        env_str: str = "room_env:RoomEnv-v2",
        env_config: dict = {
            "question_prob": 1.0,
            "seed": 42,
            "terminates_at": 99,
            "randomize_observations": True,
            "room_size": "dev",
        },
        mm_policy: str = "generalize",
        qa_policy: str = "episodic_semantic",
        explore_policy: str = "avoid_walls",
        num_samples_for_results: int = 10,
        capacity: dict = {
            "episodic": 16,
            "semantic": 16,
            "short": 16,
        },
        pretrain_semantic: bool = False,
    ) -> None:
        """Initialize the agent.

        Args
        ----
        env_str: This has to be "room_env:RoomEnv-v2"
        env_config: The configuration of the environment.
        mm_policy: Memory management policy. Choose one of "random" or
            "generalize"
        qa_policy: question answering policy Choose one of "episodic_semantic" or
            "random"
        explore_policy: The room exploration policy. Choose one of "random" or
            "avoid_walls"
        num_samples_for_results: The number of samples to validate / test the agent.
        capacity: The capacity of each human-like memory systems.
        pretrain_semantic: Whether or not to pretrain the semantic memory system.
        """
        params_to_save = deepcopy(locals())
        del params_to_save["self"]

        self.env_str = env_str
        self.env_config = env_config
        self.mm_policy = mm_policy
        assert self.mm_policy in [
            "random",
            "episodic",
            "semantic",
            "generalize",
            "rl",
            "neural",
        ]
        self.qa_policy = qa_policy
        assert self.qa_policy in [
            "episodic_semantic",
            "episodic",
            "semantic",
            "random",
            "neural",
        ]
        self.explore_policy = explore_policy
        assert self.explore_policy in ["random", "avoid_walls", "rl", "neural"]
        self.num_samples_for_results = num_samples_for_results
        self.capacity = capacity
        self.pretrain_semantic = pretrain_semantic
        self.env = gym.make(self.env_str, **self.env_config)
        self.max_total_rewards = self.env_config["terminates_at"] + 1
        self._create_directory(params_to_save)

    def _create_directory(self, params_to_save: dict) -> None:
        """Create the directory to store the results."""
        if "RoomEnv2" in os.listdir():
            self.default_root_dir = (
                f"./RoomEnv2/training_results/{str(datetime.datetime.now())}"
            )
        else:
            self.default_root_dir = f"./training_results/{str(datetime.datetime.now())}"
        os.makedirs(self.default_root_dir, exist_ok=True)
        write_yaml(params_to_save, os.path.join(self.default_root_dir, "train.yaml"))

    def remove_results_from_disk(self) -> None:
        """Remove the results from the disk."""
        shutil.rmtree(self.default_root_dir)

    def test(self):
        """Test the agent. There is no training for this agent, since it is
        handcrafted."""
        self.scores = []

        for _ in range(self.num_samples_for_results):
            score = 0
            env_started = False
            action_pair = (None, None)
            done = False
            self.init_memory_systems()

            while not done:
                if env_started:
                    (
                        (observations, question),
                        reward,
                        done,
                        truncated,
                        info,
                    ) = self.env.step(action_pair)
                    score += reward
                    if done:
                        break

                else:
                    (observations, question), info = self.env.reset()
                    env_started = True

                encode_observation(self.memory_systems, observations[0])
                assert self.memory_systems.short.get_oldest_memory()[0] == "agent"
                manage_memory(self.memory_systems, "agent", split_possessive=False)
                for obs in observations[1:]:
                    encode_observation(self.memory_systems, obs)
                    assert self.memory_systems.short.get_oldest_memory()[0] != "agent"
                    manage_memory(
                        self.memory_systems,
                        self.mm_policy,
                        split_possessive=False,
                    )

                action_qa = str(
                    answer_question(
                        self.memory_systems,
                        self.qa_policy,
                        question,
                        split_possessive=False,
                    )
                )
                action_explore = explore(self.memory_systems, self.explore_policy)
                action_pair = (action_qa, action_explore)
            self.scores.append(score)

        results = {
            "test_score": {
                "mean": round(np.mean(self.scores).item(), 2),
                "std": round(np.std(self.scores).item(), 2),
            }
        }
        write_yaml(results, os.path.join(self.default_root_dir, "results.yaml"))
        write_yaml(
            self.memory_systems.return_as_a_dict_list(),
            os.path.join(self.default_root_dir, "last_memory_state.yaml"),
        )

    def init_memory_systems(self) -> None:
        """Initialize the agent's memory systems. This has nothing to do with the
        replay buffer."""
        self.memory_systems = MemorySystems(
            episodic=EpisodicMemory(
                capacity=self.capacity["episodic"], remove_duplicates=False
            ),
            episodic_agent=EpisodicMemory(
                capacity=self.capacity["episodic"], remove_duplicates=False
            ),
            semantic=SemanticMemory(capacity=self.capacity["semantic"]),
            short=ShortMemory(capacity=self.capacity["short"]),
        )

        if self.pretrain_semantic is not None:
            assert self.capacity["semantic"] > 0
            room_layout = self.env.return_room_layout()
            _ = self.memory_systems.semantic.pretrain_semantic(
                semantic_knowledge=room_layout,
                return_remaining_space=False,
                freeze=False,
            )
