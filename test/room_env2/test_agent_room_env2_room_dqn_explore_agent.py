import logging

logger = logging.getLogger()
logger.disabled = True

import random
import unittest

import gymnasium as gym
import numpy as np
from tqdm.auto import tqdm

from RoomEnv2.agent import DQNExploreAgent, DQNMMAgent


class DQNExploreAgenttTest(unittest.TestCase):
    def test_agent(self) -> None:
        for pretrain_semantic in [False, True]:
            for test_seed in [42]:
                for ddqn in [False, True]:
                    for dueling_dqn in [False, True]:
                        # parameters
                        all_params = {
                            "env_str": "room_env:RoomEnv-v2",
                            "max_epsilon": 1.0,
                            "min_epsilon": 0.1,
                            "epsilon_decay_until": 10 * 2,
                            "gamma": 0.65,
                            "capacity": {
                                "episodic": 4,
                                "episodic_agent": 4,
                                "semantic": 4,
                                "short": 1,
                            },
                            "nn_params": {
                                "hidden_size": 4,
                                "num_layers": 2,
                                "embedding_dim": 4,
                                "v1_params": None,
                                "v2_params": {},
                                "memory_of_interest": [
                                    "episodic",
                                    "episodic_agent",
                                    "semantic",
                                    "short",
                                ],
                            },
                            "num_iterations": 10 * 2,
                            "replay_buffer_size": 16,
                            "warm_start": 16,
                            "batch_size": 4,
                            "target_update_rate": 10,
                            "pretrain_semantic": pretrain_semantic,
                            "run_test": True,
                            "num_samples_for_results": 3,
                            "train_seed": test_seed + 5,
                            "plotting_interval": 10,
                            "device": "cpu",
                            "test_seed": test_seed,
                            "mm_policy": "generalize",
                            "qa_policy": "episodic_semantic",
                            "env_config": {
                                "question_prob": 1.0,
                                "terminates_at": 9,
                                "room_size": "dev",
                            },
                            "ddqn": ddqn,
                            "dueling_dqn": dueling_dqn,
                        }
                        agent = DQNExploreAgent(**all_params)
                        agent.train()
                        agent.remove_results_from_disk()


class DQNMMAgenttTest(unittest.TestCase):
    def test_agent(self) -> None:
        for pretrain_semantic in [False, True]:
            for test_seed in [42]:
                for ddqn in [False, True]:
                    for dueling_dqn in [False, True]:
                        for split_reward_training in [True, False]:
                            # parameters
                            all_params = {
                                "env_str": "room_env:RoomEnv-v2",
                                "max_epsilon": 1.0,
                                "min_epsilon": 0.1,
                                "epsilon_decay_until": 10 * 2,
                                "gamma": 0.65,
                                "capacity": {
                                    "episodic": 4,
                                    "episodic_agent": 4,
                                    "semantic": 4,
                                    "short": 1,
                                },
                                "nn_params": {
                                    "hidden_size": 4,
                                    "num_layers": 2,
                                    "embedding_dim": 4,
                                    "v1_params": None,
                                    "v2_params": {},
                                    "memory_of_interest": [
                                        "episodic",
                                        "semantic",
                                        "short",
                                    ],
                                },
                                "num_iterations": 10 * 2,
                                "replay_buffer_size": 16,
                                "warm_start": 16,
                                "batch_size": 4,
                                "target_update_rate": 10,
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
                                    "room_size": "dev",
                                },
                                "ddqn": ddqn,
                                "dueling_dqn": dueling_dqn,
                                "split_reward_training": split_reward_training,
                            }
                            agent = DQNMMAgent(**all_params)
                            agent.train()
                            agent.remove_results_from_disk()
