import logging

logger = logging.getLogger()
logger.disabled = True

import random
import unittest

import gymnasium as gym
import numpy as np
from tqdm.auto import tqdm

from RoomEnv2.agent import DQNAgent


class RLAgentTest(unittest.TestCase):
    def test_agent(self) -> None:
        for pretrain_semantic in [True, False]:
            for test_seed in [0, 1]:
                # parameters
                all_params = {
                    "env_str": "room_env:RoomEnv-v2",
                    "max_epsilon": 1.0,
                    "min_epsilon": 0.1,
                    "epsilon_decay_until": 100 * 2,
                    "gamma": 0.99,
                    "capacity": {"episodic": 16, "semantic": 16, "short": 1},
                    "nn_params": {
                        "hidden_size": 4,
                        "num_layers": 2,
                        "n_actions": 5,
                        "embedding_dim": 4,
                        "v1_params": None,
                        "v2_params": {},
                    },
                    "num_iterations": 100 * 2,
                    "replay_buffer_size": 128,
                    "warm_start": 128,
                    "batch_size": 2,
                    "target_update_rate": 10,
                    "pretrain_semantic": pretrain_semantic,
                    "run_validation": True,
                    "run_test": True,
                    "num_samples_for_results": 10,
                    "train_seed": test_seed + 5,
                    "plotting_interval": 10,
                    "device": "cpu",
                    "test_seed": test_seed,
                    "mm_policy": "rl",
                    "qa_policy": "episodic_semantic",
                    "explore_policy": "avoid_walls",
                    "env_config": {
                        "question_prob": 1.0,
                        "seed": None,
                        "terminates_at": 99,
                        "room_size": "dev",
                    },
                }

                agent = DQNAgent(**all_params)
                agent.train()
                agent.remove_results_from_disk()
