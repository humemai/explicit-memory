import logging

logger = logging.getLogger()
logger.disabled = True

import matplotlib

matplotlib.use("Agg")

import random
import unittest

import gymnasium as gym
import numpy as np
from tqdm.auto import tqdm

from AgentRoomEnv2.agent import DQNMMAgent


class DQNMMAgentTest(unittest.TestCase):
    def test_agent(self) -> None:
        num_runs = 0
        for pretrain_semantic in [False, "include_walls"]:
            for test_seed in [42]:
                for ddqn in [False, True]:
                    for dueling_dqn in [False, True]:
                        for split_reward_training in [True, False]:
                            for episodic_agent_capacity in [0, 4]:
                                for semantic_map_capacity in [0, 4]:
                                    rng = random.Random(num_runs)
                                    # parameters
                                    capacity = {
                                        "episodic": 4,
                                        "episodic_agent": episodic_agent_capacity,
                                        "semantic": 4,
                                        "semantic_map": semantic_map_capacity,
                                        "short": 1,
                                    }
                                    memory_of_interest = [
                                        "episodic",
                                        "semantic",
                                        "short",
                                    ]
                                    if capacity["episodic_agent"] > 0:
                                        memory_of_interest.append("episodic_agent")
                                    if capacity["semantic_map"] > 0:
                                        memory_of_interest.append("semantic_map")

                                    all_params = {
                                        "env_str": "room_env:RoomEnv-v2",
                                        "max_epsilon": 1.0,
                                        "min_epsilon": 0.1,
                                        "epsilon_decay_until": 10 * 2,
                                        "gamma": 0.65,
                                        "capacity": capacity,
                                        "nn_params": {
                                            "architecture": "lstm",
                                            "hidden_size": 4,
                                            "num_layers": 2,
                                            "embedding_dim": 4,
                                            "v1_params": None,
                                            "v2_params": {},
                                            "memory_of_interest": memory_of_interest,
                                            "include_positional_encoding": True,
                                            "max_timesteps": 100,
                                            "max_strength": 100,
                                        },
                                        "num_iterations": 10 * 2,
                                        "replay_buffer_size": 16,
                                        "warm_start": 16,
                                        "batch_size": 4,
                                        "target_update_interval": 10,
                                        "pretrain_semantic": pretrain_semantic,
                                        "run_test": True,
                                        "num_samples_for_results": 3,
                                        "train_seed": test_seed + 5,
                                        "plotting_interval": 10,
                                        "device": "cpu",
                                        "test_seed": test_seed,
                                        "qa_policy": "episodic_semantic",
                                        "explore_policy": "avoid_walls",
                                        "env_config": {
                                            "question_prob": 1.0,
                                            "terminates_at": 9,
                                            "room_size": rng.choice(["xxs"]),
                                        },
                                        "ddqn": ddqn,
                                        "dueling_dqn": dueling_dqn,
                                        "split_reward_training": split_reward_training,
                                    }
                                    agent = DQNMMAgent(**all_params)
                                    agent.train()
                                    agent.remove_results_from_disk()
