from agent import HandcraftedAgent
from tqdm.auto import tqdm
import numpy as np
import random
import logging
from room_env.envs.room2 import RoomEnv2

logger = logging.getLogger()
logger.disabled = True


room_size = "foo"

capacity = {
    "episodic": 1,
    "episodic_agent": 0,
    "semantic": 1,
    "semantic_map": 0,
    "short": 1,
}

config = {
    "question_prob": 1.0,
    "terminates_at": 9,
    "randomize_observations": True,
    "room_size": room_size,
    "rewards": {"correct": 1, "wrong": -1, "partial": -1},
    "make_everything_static": False,
    "num_total_questions": 10,
    "question_interval": 1,
}


from agent.dqn import DQNMMAgent
from tqdm.auto import tqdm
import random
import logging

logger = logging.getLogger()
logger.disabled = True


params = {
    "env_str": "room_env:RoomEnv-v2",
    "max_epsilon": 1.0,
    "min_epsilon": 0.1,
    "epsilon_decay_until": 10 * 20,
    "gamma": 0,
    "capacity": capacity,
    "nn_params": {
        "hidden_size": 128,
        "num_layers": 2,
        "embedding_dim": 64,
        "v1_params": None,
        "v2_params": {},
        "memory_of_interest": [
            "episodic",
            "semantic",
            "short",
        ],
    },
    "num_iterations": 10 * 20,
    "replay_buffer_size": 128,
    "warm_start": 128,
    "batch_size": 128,
    "target_update_interval": 10,
    "pretrain_semantic": False,
    "run_test": True,
    "num_samples_for_results": 10,
    "train_seed": 5,
    "plotting_interval": 10,
    "device": "cpu",
    "test_seed": 0,
    "qa_policy": "episodic_semantic",
    "explore_policy": "avoid_walls",
    "env_config": config,
    "ddqn": True,
    "dueling_dqn": True,
    "split_reward_training": True,
    "default_root_dir": "./training_results/TRASH",
}

agent = DQNMMAgent(**params)
agent.train()
