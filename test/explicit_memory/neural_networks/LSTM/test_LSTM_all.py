import unittest

import torch
import numpy as np

from explicit_memory.nn import LSTM


class LSTMTest(unittest.TestCase):
    def test_all(self) -> None:
        configs = []

        for hidden_size in [16, 32]:
            for num_layers in [1, 2]:
                for num_actions in [2, 3]:
                    for embedding_dim in [4, 8]:
                        for capacity in [4, 8]:
                            for include_human in [None, "sum"]:
                                for batch_first in [True, False]:
                                    for human_embedding_on_object_location in [
                                        True,
                                        False,
                                    ]:
                                        for dueling_dqn in [True, False]:
                                            for fuse_information in ["concat", "sum"]:
                                                configs.append(
                                                    {
                                                        "hidden_size": hidden_size,
                                                        "num_layers": num_layers,
                                                        "n_actions": num_actions,
                                                        "embedding_dim": embedding_dim,
                                                        "capacity": {
                                                            "episodic": capacity // 2,
                                                            "semantic": capacity // 2,
                                                            "short": capacity // 2,
                                                        },
                                                        "memory_of_interest": [
                                                            "episodic",
                                                            "semantic",
                                                            "short",
                                                        ],
                                                        "entities": [
                                                            "Foo",
                                                            "Bar",
                                                            "laptop",
                                                            "phone",
                                                            "desk",
                                                            "lap",
                                                        ],
                                                        "relations": [],
                                                        "v1_params": {
                                                            "include_human": include_human,
                                                            "human_embedding_on_object_location": human_embedding_on_object_location,
                                                        },
                                                        "v2_params": None,
                                                        "batch_first": batch_first,
                                                        "device": "cpu",
                                                        "dueling_dqn": dueling_dqn,
                                                        "fuse_information": fuse_information,
                                                        "include_positional_encoding": False,
                                                    }
                                                )
        for config in configs:
            lstm = LSTM(**config)

    def test_forward_v1(self) -> None:
        for fuse_information in ["concat", "sum"]:
            config = {
                "hidden_size": 64,
                "num_layers": 2,
                "n_actions": 3,
                "embedding_dim": 32,
                "capacity": {
                    "episodic": 16,
                    "semantic": 16,
                    "short": 1,
                },
                "memory_of_interest": [
                    "episodic",
                    "semantic",
                    "short",
                ],
                "entities": [
                    "Foo",
                    "Bar",
                    "laptop",
                    "phone",
                    "desk",
                    "lap",
                ],
                "relations": [],
                "v1_params": {
                    "include_human": "sum",
                    "human_embedding_on_object_location": False,
                },
                "v2_params": None,
                "batch_first": True,
                "device": "cpu",
                "dueling_dqn": True,
                "fuse_information": fuse_information,
                "include_positional_encoding": False,
            }
            lstm = LSTM(**config)
            lstm.forward(
                np.array(
                    [
                        {
                            "episodic": [["Foo's laptop", "atlocation", "desk", 0]],
                            "semantic": [["laptop", "atlocation", "desk", 1]],
                            "short": [["Bar's phone", "atlocation", "lap", 1]],
                        }
                    ]
                )
            )

    def test_forward_v2(self) -> None:
        for fuse_information in ["concat", "sum"]:
            if fuse_information == "concat":
                include_positional_encoding = False
                max_timesteps = None
                max_strength = None
            else:
                include_positional_encoding = True
                max_timesteps = 100
                max_strength = 100
            config = {
                "hidden_size": 64,
                "num_layers": 2,
                "n_actions": 3,
                "embedding_dim": 64,
                "capacity": {
                    "episodic": 16,
                    "semantic": 16,
                    "short": 1,
                },
                "memory_of_interest": [
                    "episodic",
                    "semantic",
                    "short",
                ],
                "entities": [
                    "laptop",
                    "phone",
                    "desk",
                    "lap",
                ],
                "relations": ["atlocation"],
                "v1_params": None,
                "v2_params": {},
                "batch_first": True,
                "device": "cpu",
                "dueling_dqn": True,
                "fuse_information": fuse_information,
                "include_positional_encoding": include_positional_encoding,
                "max_timesteps": max_timesteps,
                "max_strength": max_strength,
            }
            lstm = LSTM(**config)
            lstm.forward(
                np.array(
                    [
                        {
                            "episodic": [["laptop", "atlocation", "desk", 0]],
                            "semantic": [["laptop", "atlocation", "desk", 1]],
                            "short": [["phone", "atlocation", "lap", 1]],
                        }
                    ]
                )
            )

    def test_make_categorical_embeddings(self) -> None:
        for make_categorical_embeddings in [True, False]:
            config = {
                "hidden_size": 4,
                "num_layers": 2,
                "n_actions": 3,
                "embedding_dim": 4,
                "make_categorical_embeddings": make_categorical_embeddings,
                "capacity": {
                    "episodic": 16,
                    "semantic": 16,
                    "short": 1,
                },
                "memory_of_interest": [
                    "episodic",
                    "semantic",
                    "short",
                ],
                "entities": {"c0": ["a0", "a1"], "c1": ["b0", "b1", "b2"]},
                "relations": ["r0", "r1"],
                "v1_params": None,
                "v2_params": {},
                "batch_first": True,
                "device": "cpu",
                "dueling_dqn": True,
                "fuse_information": "sum",
                "include_positional_encoding": False,
                "max_timesteps": None,
                "max_strength": None,
                
            }
            lstm = LSTM(**config)
            self.assertTrue(
                all(lstm.embeddings.weight.data[0] == torch.tensor([0, 0, 0, 0]))
            )
            if make_categorical_embeddings:
                self.assertTrue(
                    all(
                        lstm.embeddings.weight.data[1] == lstm.embeddings.weight.data[2]
                    )
                )
                self.assertTrue(
                    all(
                        lstm.embeddings.weight.data[3] == lstm.embeddings.weight.data[4]
                    )
                )
                self.assertTrue(
                    all(
                        lstm.embeddings.weight.data[3] == lstm.embeddings.weight.data[5]
                    )
                )
            else:
                self.assertFalse(
                    all(
                        lstm.embeddings.weight.data[1] == lstm.embeddings.weight.data[2]
                    )
                )
                self.assertFalse(
                    all(
                        lstm.embeddings.weight.data[3] == lstm.embeddings.weight.data[4]
                    )
                )
                self.assertFalse(
                    all(
                        lstm.embeddings.weight.data[3] == lstm.embeddings.weight.data[5]
                    )
                )
