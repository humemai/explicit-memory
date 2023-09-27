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
    SemanticMemory,
    ShortMemory,
    MemorySystems,
)
from explicit_memory.nn import LSTM
from explicit_memory.policy import (
    answer_question,
    encode_observation,
    manage_memory,
    explore,
)
from explicit_memory.utils import ReplayBuffer, is_running_notebook, write_yaml


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
        },
        memory_management_policy: str = "generalize",
        qa_policy: str = "episodic_semantic",
        explore_policy: str = "random",
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
        memory_management_policy: Memory management policy. Choose one of "random" or
            "generalize"
        qa_policy: question answering policy Choose one of "episodic_semantic" or
            "random"
        explore_policy: The room exploration policy. Choose one of "random" or
            "avoid_walls"
        num_samples_for_results: The number of samples to validate / test the agent.
        capacity: The capacity of each human-like memory systems.
        pretrain_semantic: Whether or not to pretrain the semantic memory system.

        """
        self.all_params = deepcopy(locals())
        del self.all_params["self"]
        self.env_str = env_str
        self.env_config = env_config
        self.memory_management_policy = memory_management_policy
        assert self.memory_management_policy in ["random", "generalize", "rl", "neural"]
        self.qa_policy = qa_policy
        assert self.qa_policy in ["episodic_semantic", "random", "rl", "neural"]
        self.explore_policy = explore_policy
        assert self.explore_policy in ["random", "avoid_walls", "rl", "neural"]
        self.num_samples_for_results = num_samples_for_results
        self.capacity = capacity
        self.pretrain_semantic = pretrain_semantic

        self.env = gym.make(self.env_str, **env_config)

        if "RoomEnv2" in os.listdir():
            self.default_root_dir = (
                f"./RoomEnv2/training_results/{str(datetime.datetime.now())}"
            )
        else:
            self.default_root_dir = f"./training_results/{str(datetime.datetime.now())}"
        os.makedirs(self.default_root_dir, exist_ok=True)

        self.max_total_rewards = self.env_config["terminates_at"] + 1

    def remove_results_from_disk(self) -> None:
        """Remove the results from the disk."""
        shutil.rmtree(self.default_root_dir)

    def _encode_observation(self, obs: List[str]) -> None:
        """Encode an observation into a short-term memory.

        Args
        ----
        obs: observation as a quadruple: [head, relation, tail, num]

        """
        encode_observation(self.memory_systems, obs)

    def _manage_memory(
        self, dont_generalize_agent: bool = True, split_possessive: bool = False
    ) -> None:
        """Manage the memory systems."""
        if self.memory_management_policy == "random":
            selected_action = random.choice(["episodic", "semantic", "forget"])
            manage_memory(
                self.memory_systems,
                selected_action,
                dont_generalize_agent,
                split_possessive,
            )
        elif self.memory_management_policy == "generalize":
            manage_memory(
                self.memory_systems,
                "generalize",
                dont_generalize_agent,
                split_possessive,
            )
        elif self.memory_management_policy == "neural":
            raise NotImplementedError
        else:
            raise ValueError("Unknown memory management policy.")

    def _answer_question(
        self, question: List[str], split_possessive: bool = False
    ) -> str:
        """Answer the question."""
        return str(
            answer_question(
                self.memory_systems, self.qa_policy, question, split_possessive
            )
        )

    def _explore(self) -> str:
        """Explore the room."""
        return explore(
            self.memory_systems, self.explore_policy, self.memory_management_policy
        )

    def _test(self) -> int:
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

            for obs in observations:
                self._encode_observation(obs)
                self._manage_memory()
            action_qa = self._answer_question(question)
            action_explore = explore(
                self.memory_systems, self.explore_policy, self.memory_management_policy
            )
            action_pair = (action_qa, action_explore)

        return score

    def test(self):
        """Test the agent. There is no training for this agent, since it is
        handcrafted."""
        self.scores = []

        for _ in range(self.num_samples_for_results):
            score = self._test()
            self.scores.append(score)

        results = {
            "test_score": {
                "mean": round(np.mean(self.scores).item(), 2),
                "std": round(np.std(self.scores).item(), 2),
            }
        }
        write_yaml(results, os.path.join(self.default_root_dir, "results.yaml"))
        write_yaml(self.all_params, os.path.join(self.default_root_dir, "train.yaml"))
        write_yaml(
            self.memory_systems.return_as_a_dict_list(),
            os.path.join(self.default_root_dir, "last_memory_state.yaml"),
        )

    def init_memory_systems(self) -> None:
        """Initialize the agent's memory systems. This has nothing to do with the
        replay buffer."""
        self.memory_systems = MemorySystems(
            episodic=EpisodicMemory(capacity=self.capacity["episodic"]),
            semantic=SemanticMemory(capacity=self.capacity["semantic"]),
            short=ShortMemory(capacity=self.capacity["short"]),
        )

        if self.pretrain_semantic:
            assert self.capacity["semantic"] > 0
            _ = self.memory_systems.semantic.pretrain_semantic(
                semantic_knowledge=self.env.env.room_layout,
                return_remaining_space=False,
                freeze=False,
            )
