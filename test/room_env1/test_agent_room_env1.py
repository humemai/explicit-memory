import logging

logger = logging.getLogger()
logger.disabled = True

import random
import unittest

import gymnasium as gym
from tqdm.auto import tqdm

from AgentRoomEnv1.agent import DQNAgent, HandcraftedAgent


class HandcraftedAgentTest(unittest.TestCase):
    def test_all_agents(self) -> None:
        for policy in tqdm(["random", "episodic_only", "semantic_only"]):
            for test_seed in [42]:
                all_params = {
                    "env_str": "room_env:RoomEnv-v1",
                    "env_config": {
                        "des_size": "l",
                        "question_prob": 1.0,
                        "allow_random_human": True,
                        "allow_random_question": True,
                        "check_resources": True,
                        "seed": test_seed,
                    },
                    "policy": policy,
                    "num_samples_for_results": 3,
                }
                if policy == "random":
                    all_params["capacity"] = {
                        "episodic": 16,
                        "semantic": 16,
                        "short": 1,
                    }
                elif policy == "episodic_only":
                    all_params["capacity"] = {"episodic": 32, "semantic": 0, "short": 1}
                else:
                    all_params["capacity"] = {"episodic": 0, "semantic": 32, "short": 1}

                agent = HandcraftedAgent(**all_params)
                agent.test()
                agent.remove_results_from_disk()


class RLAgentTest(unittest.TestCase):
    def test_agent(self) -> None:
        for pretrain_semantic in [False, True]:
            for test_seed in [42]:
                # parameters
                all_params = {
                    "env_str": "room_env:RoomEnv-v1",
                    "max_epsilon": 1.0,
                    "min_epsilon": 0.1,
                    "epsilon_decay_until": 128 * 1,
                    "gamma": 0.65,
                    "capacity": {"episodic": 4, "semantic": 5, "short": 1},
                    "nn_params": {
                        "hidden_size": 4,
                        "num_layers": 2,
                        "n_actions": 3,
                        "embedding_dim": 4,
                    },
                    "num_iterations": 128 * 2,
                    "replay_buffer_size": 2 * 4,
                    "warm_start": 2 * 4,
                    "batch_size": 2,
                    "target_update_interval": 10,
                    "pretrain_semantic": pretrain_semantic,
                    "run_test": True,
                    "num_samples_for_results": 3,
                    "train_seed": test_seed + 5,
                    "plotting_interval": 10,
                    "device": "cpu",
                    "test_seed": test_seed,
                }

                agent = DQNAgent(**all_params)
                agent.train()
                agent.remove_results_from_disk()
