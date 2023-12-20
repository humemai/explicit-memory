import matplotlib

matplotlib.use("Agg")

import logging

logger = logging.getLogger()
logger.disabled = True

import random

from copy import deepcopy
from tqdm.auto import tqdm
from agent.dqn import DQNMMAgent

for room_size in ["m", "l"]:
    for test_seed in [0, 1, 2, 3, 4]:
        for split_reward_training in [False, True]:
            for pretrain_semantic in [False, "exclude_walls"]:
                for randomize in ["none", "objects", "objects_middle", "all"]:
                    for gamma in [0.9, 0.99]:
                        if room_size == "m":
                            capacity = {
                                "episodic": 8,
                                "episodic_agent": 0,
                                "semantic": 8,
                                "semantic_map": 0,
                                "short": 1,
                            }
                        else:
                            capacity = {
                                "episodic": 16,
                                "episodic_agent": 0,
                                "semantic": 16,
                                "semantic_map": 0,
                                "short": 1,
                            }

                        config = {
                            "question_prob": 1.0,
                            "terminates_at": 99,
                            "randomize_observations": randomize,
                            "room_size": room_size,
                            "rewards": {"correct": 1, "wrong": -1, "partial": -1},
                            "make_everything_static": False,
                            "num_total_questions": 100,
                            "question_interval": 1,
                            "include_walls_in_observations": True,
                        }

                        params = {
                            "env_str": "room_env:RoomEnv-v2",
                            "max_epsilon": 1.0,
                            "min_epsilon": 0.1,
                            "epsilon_decay_until": 100 * 100,
                            "gamma": gamma,
                            "capacity": capacity,
                            "nn_params": {
                                "hidden_size": 64,
                                "num_layers": 2,
                                "embedding_dim": 64,
                                "v1_params": None,
                                "v2_params": {},
                                "memory_of_interest": [
                                    "episodic",
                                    "semantic",
                                    "short",
                                ],
                                "fuse_information": "sum",
                                "include_positional_encoding": True,
                                "max_timesteps": config["terminates_at"] + 1,
                                "max_strength": config["terminates_at"] + 1,
                            },
                            "num_iterations": 100 * 100,
                            "replay_buffer_size": 100 * 100,
                            "warm_start": 100 * 100 / 10,
                            "batch_size": 32,
                            "target_update_interval": 10,
                            "pretrain_semantic": pretrain_semantic,
                            "run_test": True,
                            "num_samples_for_results": 10,
                            "train_seed": test_seed + 5,
                            "plotting_interval": 10,
                            "device": "cpu",
                            "test_seed": test_seed,
                            "qa_policy": "episodic_semantic",
                            "explore_policy": "avoid_walls",
                            "env_config": config,
                            "ddqn": True,
                            "dueling_dqn": True,
                            "split_reward_training": split_reward_training,
                            "default_root_dir": f"./training_results/room_size={room_size}/split_reward_training={split_reward_training}/pretrain_semantic={pretrain_semantic}/randomize={randomize}/gamma={gamma}/",
                            "run_handcrafted_baselines": [
                                {
                                    "mm": mm,
                                    "qa": qa,
                                    "explore": explore,
                                    "pretrain_semantic": pretrain_semantic,
                                }
                                for mm in ["random", "episodic", "semantic"]
                                for qa in ["episodic_semantic"]
                                for explore in ["random", "avoid_walls"]
                                for pretrain_semantic in [False, "exclude_walls"]
                            ],
                        }
                        agent = DQNMMAgent(**params)
                        agent.train()
