import logging

logger = logging.getLogger()
logger.disabled = True

import random
import unittest

import gymnasium as gym
import numpy as np
from tqdm.auto import tqdm

from RoomEnv2.agent import HandcraftedAgent


class HandcraftedAgentTest(unittest.TestCase):
    def test_all_agents(self) -> None:
        capacity = {"episodic": 16, "semantic": 16, "short": 1}

        config = {
            "question_prob": 1.0,
            "seed": 42,
            "terminates_at": 99,
        }

        results = {}
        for mm_policy in ["random", "generalize"]:
            for qa_policy in ["random", "episodic_semantic"]:
                for explore_policy in ["random", "avoid_walls"]:
                    for pretrain_semantic in [False, True]:
                        key = (mm_policy, qa_policy, explore_policy)
                        if key not in results:
                            results[key] = []
                        print(
                            mm_policy,
                            qa_policy,
                            explore_policy,
                        )

                        for seed in tqdm([0, 1]):
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
                            to_append = (np.mean(agent.scores), np.std(agent.scores))
                            print(to_append)
                            results[key].append(to_append)
