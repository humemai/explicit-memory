import unittest

import numpy as np
import torch

from explicit_memory.nn import LSTM


class LSTMTest(unittest.TestCase):
    def test_v1_fuse_concat(self) -> None:
        config = {
            "hidden_size": 16,
            "num_layers": 2,
            "n_actions": 3,
            "embedding_dim": 16,
            "capacity": {
                "episodic": 4,
                "semantic": 4,
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
            "fuse_information": "concat",
            "include_positional_encoding": False,
        }
        lstm = LSTM(**config)
        self.assertEqual(lstm.linear_layer_hidden_size, 48)
        self.assertEqual(lstm.embeddings.num_embeddings, 7)
        self.assertEqual(lstm.embeddings.embedding_dim, 16)
        self.assertEqual(lstm.embeddings.padding_idx, 0)
        self.assertEqual(lstm.input_size_s, 32)
        self.assertEqual(lstm.input_size_e, 32)
        self.assertEqual(lstm.input_size_o, 32)

        q_values = lstm.forward(
            np.array(
                [
                    {
                        "episodic": [["Foo's laptop", "atlocation", "desk", 0]],
                        "semantic": [["laptop", "atlocation", "desk", 1]],
                        "short": [["Bar's phone", "atlocation", "lap", 1]],
                    },
                    {
                        "episodic": [
                            ["Foo's laptop", "atlocation", "desk", 0],
                            ["Bar's laptop", "atlocation", "desk", 0],
                        ],
                        "semantic": [["laptop", "atlocation", "desk", 1]],
                        "short": [["Bar's phone", "atlocation", "lap", 1]],
                    },
                ]
            )
        )
        self.assertEqual(q_values.shape, torch.Size([2, 3]))

    def test_v1_fuse_sum(self) -> None:
        config = {
            "hidden_size": 16,
            "num_layers": 2,
            "n_actions": 3,
            "embedding_dim": 16,
            "capacity": {
                "episodic": 4,
                "semantic": 4,
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
            "fuse_information": "sum",
            "include_positional_encoding": False,
        }
        lstm = LSTM(**config)
        self.assertEqual(lstm.linear_layer_hidden_size, 16)
        self.assertEqual(lstm.embeddings.num_embeddings, 7)
        self.assertEqual(lstm.embeddings.embedding_dim, 16)
        self.assertEqual(lstm.embeddings.padding_idx, 0)
        self.assertEqual(lstm.input_size_s, 16)
        self.assertEqual(lstm.input_size_e, 16)
        self.assertEqual(lstm.input_size_o, 16)

        q_values = lstm.forward(
            np.array(
                [
                    {
                        "episodic": [["Foo's laptop", "atlocation", "desk", 0]],
                        "semantic": [["laptop", "atlocation", "desk", 1]],
                        "short": [["Bar's phone", "atlocation", "lap", 1]],
                    },
                    {
                        "episodic": [
                            ["Foo's laptop", "atlocation", "desk", 0],
                            ["Bar's laptop", "atlocation", "desk", 0],
                        ],
                        "semantic": [["laptop", "atlocation", "desk", 1]],
                        "short": [["Bar's phone", "atlocation", "lap", 1]],
                    },
                ]
            )
        )
        self.assertEqual(q_values.shape, torch.Size([2, 3]))

    def test_v1_positional_encoding(self) -> None:
        config = {
            "hidden_size": 16,
            "num_layers": 2,
            "n_actions": 3,
            "embedding_dim": 16,
            "capacity": {
                "episodic": 4,
                "semantic": 4,
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
            "fuse_information": "sum",
            "include_positional_encoding": True,
            "max_timesteps": None,
            "max_strength": 10,
        }
        with self.assertRaises(AssertionError):
            lstm = LSTM(**config)

        config["max_timesteps"] = 10
        lstm = LSTM(**config)
        self.assertEqual(lstm.linear_layer_hidden_size, 16)
        self.assertEqual(lstm.embeddings.num_embeddings, 7)
        self.assertEqual(lstm.embeddings.embedding_dim, 16)
        self.assertEqual(lstm.embeddings.padding_idx, 0)
        self.assertEqual(lstm.input_size_s, 16)
        self.assertEqual(lstm.input_size_e, 16)
        self.assertEqual(lstm.input_size_o, 16)

        q_values = lstm.forward(
            np.array(
                [
                    {
                        "episodic": [["Foo's laptop", "atlocation", "desk", 0]],
                        "semantic": [["laptop", "atlocation", "desk", 1]],
                        "short": [["Bar's phone", "atlocation", "lap", 1]],
                    },
                    {
                        "episodic": [
                            ["Foo's laptop", "atlocation", "desk", 0],
                            ["Bar's laptop", "atlocation", "desk", 0],
                        ],
                        "semantic": [["laptop", "atlocation", "desk", 1]],
                        "short": [["Bar's phone", "atlocation", "lap", 1]],
                    },
                ]
            )
        )
        self.assertEqual(q_values.shape, torch.Size([2, 3]))

    def test_v2_fuse_concat(self) -> None:
        config = {
            "hidden_size": 16,
            "num_layers": 2,
            "n_actions": 3,
            "embedding_dim": 16,
            "capacity": {
                "episodic": 4,
                "semantic": 4,
                "short": 1,
            },
            "memory_of_interest": [
                "episodic",
                "semantic",
                "short",
            ],
            "entities": ["agent", "room0", "room1", "room2", "dep0", "ind1", "wall"],
            "relations": ["atlocation", "north", "south", "east"],
            "v1_params": None,
            "v2_params": {},
            "batch_first": True,
            "device": "cpu",
            "dueling_dqn": True,
            "fuse_information": "concat",
            "include_positional_encoding": True,
            "max_timesteps": 10,
            "max_strength": 10,
        }
        lstm = LSTM(**config)
        self.assertEqual(lstm.linear_layer_hidden_size, 48)
        self.assertEqual(lstm.embeddings.num_embeddings, 12)
        self.assertEqual(lstm.embeddings.embedding_dim, 16)
        self.assertEqual(lstm.embeddings.padding_idx, 0)
        self.assertEqual(lstm.input_size_s, 48)
        self.assertEqual(lstm.input_size_e, 48)
        self.assertEqual(lstm.input_size_o, 48)

        q_values = lstm.forward(
            np.array(
                [
                    {
                        "episodic": [["agent", "atlocation", "room0", 0]],
                        "semantic": [["room0", "north", "room1", 1]],
                        "short": [["room1", "south", "room2", 8]],
                    },
                    {
                        "episodic": [["agent", "atlocation", "room0", 0]],
                        "semantic": [
                            ["room0", "north", "room1", 1],
                            ["dep0", "atlocation", "room0", 3],
                            ["ind1", "atlocation", "room2", 4],
                        ],
                        "short": [["room2", "east", "wall", 9]],
                    },
                ]
            )
        )
        self.assertEqual(q_values.shape, torch.Size([2, 3]))

    def test_v2_fuse_sum(self) -> None:
        config = {
            "hidden_size": 16,
            "num_layers": 2,
            "n_actions": 3,
            "embedding_dim": 16,
            "capacity": {
                "episodic": 4,
                "semantic": 4,
                "short": 1,
            },
            "memory_of_interest": [
                "episodic",
                "semantic",
                "short",
            ],
            "entities": ["agent", "room0", "room1", "room2", "dep0", "ind1", "wall"],
            "relations": ["atlocation", "north", "south", "east"],
            "v1_params": None,
            "v2_params": {},
            "batch_first": True,
            "device": "cpu",
            "dueling_dqn": True,
            "fuse_information": "sum",
            "include_positional_encoding": True,
            "max_timesteps": 10,
            "max_strength": 10,
        }
        lstm = LSTM(**config)
        self.assertEqual(lstm.linear_layer_hidden_size, 16)
        self.assertEqual(lstm.embeddings.num_embeddings, 12)
        self.assertEqual(lstm.embeddings.embedding_dim, 16)
        self.assertEqual(lstm.embeddings.padding_idx, 0)
        self.assertEqual(lstm.input_size_s, 16)
        self.assertEqual(lstm.input_size_e, 16)
        self.assertEqual(lstm.input_size_o, 16)

        q_values = lstm.forward(
            np.array(
                [
                    {
                        "episodic": [["agent", "atlocation", "room0", 0]],
                        "semantic": [["room0", "north", "room1", 1]],
                        "short": [["room1", "south", "room2", 8]],
                    },
                    {
                        "episodic": [["agent", "atlocation", "room0", 0]],
                        "semantic": [
                            ["room0", "north", "room1", 1],
                            ["dep0", "atlocation", "room0", 3],
                            ["ind1", "atlocation", "room2", 4],
                        ],
                        "short": [["room2", "east", "wall", 9]],
                    },
                ]
            )
        )
        self.assertEqual(q_values.shape, torch.Size([2, 3]))
