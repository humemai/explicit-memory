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

from AgentRoomEnv2.agent import DQNLSTMBaselineAgent


class DQNLSTMBaselineAgentTest(unittest.TestCase):
    def test_agent(self) -> None:
        hparams = {
            "env_str": "room_env:RoomEnv-v2",
            "num_iterations": 10,
            "replay_buffer_size": 10,
            "warm_start": 10,
            "batch_size": 2,
            "target_update_interval": 10,
            "epsilon_decay_until": 10,
            "max_epsilon": 1.0,
            "min_epsilon": 0.1,
            "gamma": 0.9,
            "history_block_size": 2,
            "nn_params": {
                "architecture": "lstm",
                "hidden_size": 4,
                "num_layers": 2,
                "embedding_dim": 4,
                "fuse_information": "sum",
                "include_positional_encoding": True,
                "max_timesteps": 100,
                "max_strength": 100,
            },
            "run_test": True,
            "num_samples_for_results": 10,
            "plotting_interval": 10,
            "train_seed": 6,
            "test_seed": 1,
            "device": "cpu",
            "env_config": {
                "question_prob": 1.0,
                "terminates_at": 4,
                "randomize_observations": "objects",
                "room_size": "l",
                "rewards": {"correct": 1, "wrong": 0, "partial": 0},
                "make_everything_static": False,
                "num_total_questions": 10,
                "question_interval": 1,
                "include_walls_in_observations": True,
            },
            "ddqn": True,
            "dueling_dqn": True,
            "default_root_dir": "./training_results/",
            "run_handcrafted_baselines": True,
        }

        agent = DQNLSTMBaselineAgent(**hparams)
        agent.train()
        agent.remove_results_from_disk()
