import datetime
import os
import random
import shutil
from copy import deepcopy

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from tqdm.auto import tqdm, trange

from explicit_memory.memory import (EpisodicMemory, MemorySystems,
                                    SemanticMemory, ShortMemory)
from explicit_memory.nn import LSTM
from explicit_memory.policy import (answer_question, encode_observation,
                                    manage_memory)
from explicit_memory.utils import ReplayBuffer, is_running_notebook, write_yaml


class HandcraftedAgent:

    """Handcrafted agent interacting with environment. This agent is not trained.
    Only one of the three agents, i.e., random, episodic_only, and semantic_only are
    suported
    """

    def __init__(
        self,
        env_str: str = "room_env:RoomEnv-v1",
        env_config: dict = {
            "des_size": "l",
            "question_prob": 1.0,
            "allow_random_human": False,
            "allow_random_question": False,
            "check_resources": True,
            "seed": 42,
        },
        policy: str = "random",
        num_samples_for_results: int = 10,
        capacity: dict = {
            "episodic": 16,
            "semantic": 16,
            "short": 1,
        },
        pretrain_semantic: str | bool = False,
        default_root_dir: str = "./training_results/",
    ) -> None:
        """Initialization.

        Args:
            env_str: This has to be "room_env:RoomEnv-v1"
            env_config: The configuration of the environment.
            policy: The memory management policy. Choose one of "random", "episodic_only",
                    or "semantic_only".
            num_samples_for_results: The number of samples to validate / test the agent.
            capacity: The capacity of each human-like memory systems.
            pretrain_semantic: Whether or not to pretrain the semantic memory system.
            default_root_dir: default root directory to store the results.
            des_size: The size of the DES. Choose one of "xxs", "xs", "s", "m", or "l".

        """
        params_to_save = deepcopy(locals())
        del params_to_save["self"]

        self.env_str = env_str
        self.policy = policy
        self.num_samples_for_results = num_samples_for_results
        self.capacity = capacity
        self.pretrain_semantic = pretrain_semantic
        self.env = gym.make(self.env_str, **env_config)
        self.default_root_dir = os.path.join(
            default_root_dir, str(datetime.datetime.now())
        )
        self._create_directory(params_to_save)

    def _create_directory(self, params_to_save: dict) -> None:
        """Create the directory to store the results."""
        os.makedirs(self.default_root_dir, exist_ok=True)
        write_yaml(params_to_save, os.path.join(self.default_root_dir, "train.yaml"))

    def remove_results_from_disk(self) -> None:
        """Remove the results from the disk."""
        shutil.rmtree(self.default_root_dir)

    def init_memory_systems(self) -> None:
        """Initialize the agent's memory systems. This has nothing to do with the
        replay buffer."""

        self.memory_systems = MemorySystems(
            episodic=EpisodicMemory(
                capacity=self.capacity["episodic"], remove_duplicates=False
            ),
            semantic=SemanticMemory(capacity=self.capacity["semantic"]),
            short=ShortMemory(capacity=self.capacity["short"]),
        )

        if self.pretrain_semantic:
            assert self.capacity["semantic"] > 0
            _ = self.memory_systems.semantic.pretrain_semantic(
                semantic_knowledge=self.env.des.semantic_knowledge,
                return_remaining_space=False,
                freeze=False,
            )

    def test(self):
        """Test the agent. There is no training for this agent, since it is
        handcrafted."""
        self.scores = []
        for _ in range(self.num_samples_for_results):
            self.init_memory_systems()
            (observation, question), info = self.env.reset()
            encode_observation(self.memory_systems, observation)

            done = False
            score = 0
            while not done:
                if self.policy.lower() == "random":
                    selected_action = random.choice(["episodic", "semantic", "forget"])
                    manage_memory(
                        self.memory_systems, selected_action, split_possessive=True
                    )
                    qa_policy = "episodic_semantic"
                elif self.policy.lower() == "episodic_only":
                    manage_memory(
                        self.memory_systems, "episodic", split_possessive=True
                    )
                    qa_policy = "episodic"
                elif self.policy.lower() == "semantic_only":
                    qa_policy = "semantic"
                    manage_memory(
                        self.memory_systems, "semantic", split_possessive=True
                    )
                else:
                    raise ValueError("Unknown policy.")
                answer = str(
                    answer_question(self.memory_systems, qa_policy, question)
                ).lower()
                (
                    (observation, question),
                    reward,
                    done,
                    truncated,
                    info,
                ) = self.env.step(answer)

                encode_observation(self.memory_systems, observation)
                score += reward
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
