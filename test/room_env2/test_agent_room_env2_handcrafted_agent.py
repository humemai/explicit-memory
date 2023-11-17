import logging

logger = logging.getLogger()
logger.disabled = True

import random
import unittest

import gymnasium as gym
import numpy as np
from tqdm.auto import tqdm

from AgentRoomEnv2.agent import HandcraftedAgent


class HandcraftedAgentTest(unittest.TestCase):
    def test_all_agents(self) -> None:
        for idx in range(10):
            rng = random.Random(idx)
            capacity = {
                "episodic": random.randint(1, 4),
                "semantic": random.randint(1, 4),
                "episodic_agent": random.randint(0, 4),
                "semantic_map": random.randint(0, 4),
                "short": 1,
            }

            config = {
                "question_prob": 1.0,
                "seed": 42,
                "terminates_at": 99,
                "room_size": rng.choice(["xxs", "xs", "s", "m", "l"]),
            }

            results = {}
            for mm_policy in ["random", "generalize"]:
                for qa_policy in ["random", "episodic_semantic"]:
                    for explore_policy in ["random", "avoid_walls"]:
                        for pretrain_semantic in [False, True]:
                            key = (mm_policy, qa_policy, explore_policy)
                            if key not in results:
                                results[key] = []

                            for seed in tqdm([42]):
                                config["seed"] = seed

                                agent = HandcraftedAgent(
                                    env_str="room_env:RoomEnv-v2",
                                    env_config=config,
                                    mm_policy=mm_policy,
                                    qa_policy=qa_policy,
                                    explore_policy=explore_policy,
                                    num_samples_for_results=3,
                                    capacity=capacity,
                                    pretrain_semantic=pretrain_semantic,
                                )
                                agent.test()
                                agent.remove_results_from_disk()
                                to_append = (
                                    np.mean(agent.scores),
                                    np.std(agent.scores),
                                )
                                print(to_append)
                                results[key].append(to_append)
