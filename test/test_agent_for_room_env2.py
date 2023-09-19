import logging

logger = logging.getLogger()
logger.disabled = True

import random
import unittest

import gymnasium as gym
import numpy as np
from tqdm.auto import tqdm

from RoomEnv2.agent import DQNAgent, HandcraftedAgent


class HandcraftedAgentTest(unittest.TestCase):
    def test_all_agents(self) -> None:
        capacity = {"episodic": 16, "semantic": 16, "short": 16}

        config = {
            "question_prob": 1.0,
            "seed": 42,
            "terminates_at": 99,
        }

        results = {}
        for memory_management_policy in ["random", "generalize"]:
            for qa_policy in ["random", "episodic_semantic"]:
                for explore_policy in ["random", "avoid_walls"]:
                    key = (memory_management_policy, qa_policy, explore_policy)
                    if key not in results:
                        results[key] = []
                    print(
                        memory_management_policy,
                        qa_policy,
                        explore_policy,
                    )

                    for seed in tqdm([0, 1]):
                        config["seed"] = seed

                        agent = HandcraftedAgent(
                            env_str="room_env:RoomEnv-v2",
                            env_config=config,
                            memory_management_policy=memory_management_policy,
                            qa_policy=qa_policy,
                            explore_policy=explore_policy,
                            num_samples_for_results=3,
                            capacity=capacity,
                        )
                        agent.test()
                        agent.remove_results_from_disk()
                        to_append = (np.mean(agent.scores), np.std(agent.scores))
                        print(to_append)
                        results[key].append(to_append)


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
                    "capacity": {"episodic": 16, "semantic": 16, "short": 16},
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
                }

                agent = DQNAgent(**all_params)
                agent.train()
                agent.remove_results_from_disk()
