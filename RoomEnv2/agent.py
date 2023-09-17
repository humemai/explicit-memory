"""Agent for the RoomEnv2 environment."""
import datetime
import os
import random
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

from explicit_memory.memory import EpisodicMemory, SemanticMemory, ShortMemory
from explicit_memory.policy import (answer_question, encode_observation,
                                    manage_memory)
from explicit_memory.utils import ReplayBuffer, is_running_notebook, write_yaml


class HandCraftedAgent:
    """Handcrafted agent interacting with environment.

    This agent explores the roooms, i.e., KGs. The exploration can be uniform-random,
    or just avoiding walls.

    """

    def __init__(
        self,
        env_str: str = "room_env:RoomEnv-v2",
        env_config: dict = None,
        memory_management_policy: str = "generalize",
        qa_policy: str = "episodic_semantic",
        explore_policy: str = "random",
        num_samples_for_results: int = 10,
        capacity: dict = {
            "episodic": 4,
            "semantic": 4,
            "short": 16,
        },
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

        """
        self.all_params = deepcopy(locals())
        del self.all_params["self"]
        self.env_str = env_str
        self.env_config = env_config
        self.memory_management_policy = memory_management_policy
        assert self.memory_management_policy in ["random", "generalize"]
        self.qa_policy = qa_policy
        assert self.qa_policy in ["episodic_semantic", "random"]
        self.explore_policy = explore_policy
        assert self.explore_policy in ["random", "avoid_walls"]
        self.num_samples_for_results = num_samples_for_results
        self.capacity = capacity

        self.env = gym.make(self.env_str, **env_config)

        self.default_root_dir = f"./training_results/{str(datetime.datetime.now())}"
        os.makedirs(self.default_root_dir, exist_ok=True)

    def init_memory_systems(self, num_actions: int = 5) -> None:
        """Initialize the agent's memory systems. This has nothing to do with the
        replay buffer."""
        self.memory_systems = {
            "episodic": EpisodicMemory(capacity=self.capacity["episodic"]),
            "semantic": SemanticMemory(capacity=self.capacity["semantic"]),
            "short": ShortMemory(capacity=self.capacity["short"]),
        }

    def get_memory_state(self) -> dict:
        """Return the current state of the memory systems. This is NOT what the gym env
        gives you. This is made by the agent.

        """
        state_as_dict = {
            "episodic": self.memory_systems["episodic"].return_as_lists(),
            "semantic": self.memory_systems["semantic"].return_as_lists(),
            "short": self.memory_systems["short"].return_as_lists(),
        }
        return state_as_dict

    def encode_all_observations(self, observations: List[List[str]]) -> None:
        """Encode all observations to the short-term memory systems.

        Args
        ----
        observations: A list of list of quadruples.

        """
        for obs in observations:
            encode_observation(self.memory_systems, obs)

    def _manage_memory(
        self, dont_generalize_agent: bool = True, split_possessive: bool = False
    ) -> None:
        """Manage the memory systems."""
        while not self.memory_systems["short"].is_empty:
            if self.memory_management_policy == "random":
                selected_action = random.choice(["episodic", "semantic", "forget"])
                manage_memory(
                    self.memory_systems,
                    selected_action,
                    dont_generalize_agent,
                    split_possessive,
                )
            elif self.memory_management_policy.lower() == "generalize":
                manage_memory(
                    self.memory_systems,
                    "generalize",
                    dont_generalize_agent,
                    split_possessive,
                )
            else:
                raise ValueError("Unknown memory management policy.")

    def _answer_question(self, question: str, split_possessive: bool = False) -> str:
        """Answer the question."""
        return str(
            answer_question(
                self.memory_systems, self.qa_policy, question, split_possessive
            )
        ).lower()

    def _explore_room(self) -> str:
        """Explore the room (sub-graph).

        Returns
        -------
        action: The exploration action to take.

        """
        if self.explore_policy == "random":
            action = random.choice(["north", "east", "south", "west", "stay"])
        elif self.explore_policy == "avoid_walls":
            if self.memory_management_policy == "generalize":
                memories_rooms = []
                MARKER = "^^^"  # to allow hashing

                for memory_type in ["episodic", "semantic"]:
                    memories_rooms += [
                        MARKER.join(entry[:-1])
                        for entry in self.get_memory_state()[memory_type]
                        if entry[1] in ["north", "east", "south", "west"]
                        and entry[2] != "wall"
                    ]

                memories_rooms = [
                    mem.split(MARKER) for mem in list(set(memories_rooms))
                ]

                assert (
                    self.memory_systems["episodic"].entries[-1][0] == "agent"
                ), f"{self.memory_systems['episodic'].entries[-1]}"
                agent_current_location = self.memory_systems["episodic"].entries[-1][2]

                memories_rooms = [
                    mem for mem in memories_rooms if mem[0] == agent_current_location
                ]

                if len(memories_rooms) == 0:
                    action = random.choice(["north", "east", "south", "west", "stay"])
                else:
                    action = random.choice(memories_rooms)[1]
            elif self.memory_management_policy == "random":
                action = random.choice(["north", "east", "south", "west", "stay"])
        else:
            raise ValueError("Unknown exploration policy.")

        return action

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

            self.encode_all_observations(observations)
            self._manage_memory()
            action_qa = self._answer_question(question)
            action_explore = self._explore_room()
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
            self.get_memory_state(),
            os.path.join(self.default_root_dir, "last_memory_state.yaml"),
        )
