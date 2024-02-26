import random
import unittest

import numpy as np

from explicit_memory.nn import LSTM
from explicit_memory.utils import *
from explicit_memory.utils.dqn import *


class ReplayBufferTest(unittest.TestCase):
    def setUp(self) -> None:
        self.obs = {
            "episodic": [
                ["tae's laptop", "atlocation", "desk", 1],
                ["tae's laptop", "atlocation", "desk", 2],
                ["vincent's laptop", "atlocation", "desk", 3],
                ["vincent's laptop", "atlocation", "desk", 4],
            ],
            "semantic": [
                ["laptop", "atlocation", "room1", 1],
                ["laptop", "atlocation", "room2", 4],
            ],
            "short": [
                ["michael's laptop", "atlocation", "desk", 5],
            ],
        }
        self.act = np.array([0])
        self.rew = np.array([1])
        self.done = False

    def test_wrong_init(self):
        with self.assertRaises(ValueError):
            ReplayBuffer(observation_type="foo", size=8, batch_size=16)

    def test_wrong_size(self):
        with self.assertRaises(ValueError):
            buffer = ReplayBuffer(observation_type="dict", size=8, batch_size=16)

    def test_sample_batch(self):
        buffer = ReplayBuffer(observation_type="dict", size=16, batch_size=8)

        for _ in range(buffer.max_size):
            buffer.store(
                obs=self.obs,
                act=self.act,
                rew=self.rew,
                next_obs=self.obs,
                done=self.done,
            )
        batch = buffer.sample_batch()
        self.assertEqual(buffer.size, 16)
        self.assertEqual(buffer.ptr, 0)
        self.assertEqual(batch["obs"].shape, (8,))
        self.assertEqual(batch["next_obs"].shape, (8,))
        self.assertEqual(batch["acts"].shape, (8,))
        self.assertEqual(batch["rews"].shape, (8,))
        self.assertEqual(batch["done"].shape, (8,))

        config = {
            "is_dqn_or_ppo": "dqn",
            "hidden_size": 64,
            "num_layers": 2,
            "n_actions": 3,
            "embedding_dim": 32,
            "capacity": {
                "episodic": 16,
                "semantic": 16,
                "short": 1,
            },
            "memory_of_interest": ["episodic", "semantic", "short"],
            "entities": [
                "Foo",
                "Bar",
                "laptop",
                "phone",
                "desk",
                "lap",
                "room1",
                "room2",
                "kitchen",
                "tae",
                "vincent",
                "michael",
            ],
            "relations": ["atlocation"],
            "v1_params": {
                "include_human": "sum",
                "human_embedding_on_object_location": False,
            },
            "v2_params": None,
            "batch_first": True,
            "device": "cpu",
            "include_positional_encoding": False,
        }
        lstm = LSTM(**config)
        q = lstm(batch["obs"])

    def test_positional_encoding(self):
        foo = positional_encoding(10, 32, return_tensor=False)
        self.assertEqual(foo.shape, (10, 32))
        self.assertEqual(type(foo), np.ndarray)

        foo = positional_encoding(10, 32, return_tensor=True)
        self.assertEqual(foo.shape, (10, 32))
        self.assertEqual(type(foo), torch.Tensor)
