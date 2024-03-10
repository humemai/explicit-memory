"""Handcrafted Agent for the RoomEnv2 environment."""

import datetime
import os
import shutil
from copy import deepcopy

import gymnasium as gym
import numpy as np
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
            "randomize_observations": "all",
            "make_everything_static": False,
            "rewards": {"correct": 1, "wrong": -1, "partial": 0},
            "num_total_questions": 100,
            "question_interval": 1,
            "room_size": "xxs",
        },
        mm_policy: str = "generalize",
        qa_policy: str = "episodic_semantic",
        explore_policy: str = "avoid_walls",
        num_samples_for_results: int = 10,
        capacity: dict = {
            "episodic": 16,
            "semantic": 16,
            "short": 1,
        },
        pretrain_semantic: str | bool = False,
        default_root_dir: str = "./training_results/",
    ) -> None:
        """Initialize the agent.

        Args:
            env_str: This has to be "room_env:RoomEnv-v2"
            env_config: The configuration of the environment.
            mm_policy: Memory management policy. Choose one of "random" or
                "generalize"
            qa_policy: question answering policy Choose one of "episodic_semantic" or
                "random"
            explore_policy: The room exploration policy. Choose one of "random",
                "avoid_walls", or "new_room"
            num_samples_for_results: The number of samples to validate / test the agent.
            capacity: The capacity of each human-like memory systems.
            pretrain_semantic: Whether or not to pretrain the semantic memory system.
            default_root_dir: default root directory to store the results.

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
        assert self.explore_policy in [
            "random",
            "avoid_walls",
            "new_room",
            "rl",
            "neural",
        ]
        self.num_samples_for_results = num_samples_for_results
        self.capacity = capacity
        self.pretrain_semantic = pretrain_semantic
        self.env = gym.make(self.env_str, **self.env_config)
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

    def manage_agent_and_map_memory(
        self, observations: list[list[str]]
    ) -> list[list[str]]:
        """Manage episodic_agent and semantic_map memories.

        Args:
            observations: The observations["room"] from the environment.

        Returns:
            observations_others: The observations["room"] - (episodic_agent and
                semantic_map memories) from the environment.

        """
        if hasattr(self.memory_systems, "episodic_agent"):
            observations_agent = [obs for obs in observations if obs[0] == "agent"]
            for obs in observations_agent:
                encode_observation(self.memory_systems, obs)
                manage_memory(
                    self.memory_systems, "episodic_agent", split_possessive=False
                )

        else:
            observations_agent = []

        if hasattr(self.memory_systems, "semantic_map"):
            observations_map = [
                obs
                for obs in observations
                if obs[1] in ["north", "east", "south", "west"] and obs[2] != "wall"
            ]

            for obs in observations_map:
                encode_observation(self.memory_systems, obs)
                manage_memory(
                    self.memory_systems, "semantic_map", split_possessive=True
                )

        else:
            observations_map = []

        observations_others = [
            obs
            for obs in observations
            if obs not in observations_map and obs not in observations_agent
        ]

        assert len(observations_agent) + len(observations_map) + len(
            observations_others
        ) == len(observations)

        return observations_others

    def test(self):
        """Test the agent. There is no training for this agent, since it is
        handcrafted."""
        self.scores = []

        for _ in range(self.num_samples_for_results):
            score = 0
            env_started = False
            action_pair = ([], None)
            done = False
            self.init_memory_systems()

            while not done:
                if env_started:
                    (
                        observations,
                        reward,
                        done,
                        truncated,
                        info,
                    ) = self.env.step(action_pair)
                    score += reward
                    if done:
                        break

                else:
                    observations, info = self.env.reset()
                    env_started = True

                observations["room"] = self.manage_agent_and_map_memory(
                    observations["room"]
                )
                for obs in observations["room"]:
                    encode_observation(self.memory_systems, obs)
                    manage_memory(
                        self.memory_systems,
                        self.mm_policy,
                        split_possessive=False,
                    )

                actions_qa = [
                    str(
                        answer_question(
                            self.memory_systems,
                            self.qa_policy,
                            question,
                            split_possessive=False,
                        )
                    )
                    for question in observations["questions"]
                ]

                action_explore = explore(self.memory_systems, self.explore_policy)
                action_pair = (actions_qa, action_explore)
            self.scores.append(score)

        self.scores = {
            "test_score": {
                "mean": round(np.mean(self.scores).item(), 2),
                "std": round(np.std(self.scores).item(), 2),
            }
        }
        write_yaml(self.scores, os.path.join(self.default_root_dir, "results.yaml"))
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
                capacity=self.capacity["episodic_agent"], remove_duplicates=False
            ),
            semantic=SemanticMemory(capacity=self.capacity["semantic"]),
            semantic_map=SemanticMemory(capacity=self.capacity["semantic_map"]),
            short=ShortMemory(capacity=self.capacity["short"]),
        )

        assert self.pretrain_semantic in [False, "exclude_walls", "include_walls"]
        if self.pretrain_semantic in ["exclude_walls", "include_walls"]:
            if self.pretrain_semantic == "exclude_walls":
                exclude_walls = True
            else:
                exclude_walls = False
            room_layout = self.env.unwrapped.return_room_layout(exclude_walls)

            if hasattr(self.memory_systems, "semantic_map"):
                assert self.capacity["semantic_map"] > 0
                _ = self.memory_systems.semantic_map.pretrain_semantic(
                    semantic_knowledge=room_layout,
                    return_remaining_space=False,
                    freeze=False,
                )
            else:
                assert self.capacity["semantic"] > 0
                _ = self.memory_systems.semantic.pretrain_semantic(
                    semantic_knowledge=room_layout,
                    return_remaining_space=False,
                    freeze=False,
                )
