import unittest

import torch
import numpy as np

from RoomEnv1.nn.lstm import LSTM


class LSTMTest(unittest.TestCase):
    # def test_all(self) -> None:
    #     configs = []

    #     for hidden_size in [16, 32, 64]:
    #         for num_layers in [1, 2, 4]:
    #             for n_actions in [2, 3]:
    #                 for embedding_dim in [32, 64]:
    #                     for capacity in [4, 8, 16, 32, 64]:
    #                         for include_human in [None, "sum", "concat"]:
    #                             for atom_size in [10, 51]:
    #                                 for support in [
    #                                     torch.linspace(0.0, 200.0, atom_size).to("cpu")
    #                                 ]:
    #                                     for batch_first in [True, False]:
    #                                         for memory_systems in [
    #                                             ["episodic"],
    #                                             ["semantic"],
    #                                             ["episodic", "semantic"],
    #                                             ["episodic", "semantic", "short"],
    #                                         ]:
    #                                             for (
    #                                                 human_embedding_on_object_location
    #                                             ) in [
    #                                                 True,
    #                                                 False,
    #                                             ]:
    #                                                 configs.append(
    #                                                     {
    #                                                         "hidden_size": hidden_size,
    #                                                         "num_layers": num_layers,
    #                                                         "n_actions": n_actions,
    #                                                         "embedding_dim": embedding_dim,
    #                                                         "capacity": {
    #                                                             "episodic": capacity
    #                                                             // 2,
    #                                                             "semantic": capacity
    #                                                             // 2,
    #                                                             "short": capacity // 2,
    #                                                         },
    #                                                         "entities": {
    #                                                             "humans": [
    #                                                                 "Foo",
    #                                                                 "Bar",
    #                                                             ],
    #                                                             "objects": [
    #                                                                 "laptop",
    #                                                                 "phone",
    #                                                             ],
    #                                                             "object_locations": [
    #                                                                 "desk",
    #                                                                 "lap",
    #                                                             ],
    #                                                         },
    #                                                         "include_human": include_human,
    #                                                         "atom_size": atom_size,
    #                                                         "support": support,
    #                                                         "batch_first": batch_first,
    #                                                         "accelerator": "cpu",
    #                                                         "memory_systems": memory_systems,
    #                                                         "human_embedding_on_object_location": human_embedding_on_object_location,
    #                                                     }
    #                                                 )
    #     for config in configs:
    #         lstm = LSTM(**config)

    def test_forward(self) -> None:
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
            "entities": {
                "humans": [
                    "Foo",
                    "Bar",
                ],
                "objects": [
                    "laptop",
                    "phone",
                ],
                "object_locations": [
                    "desk",
                    "lap",
                ],
            },
            "include_human": "sum",
            "atom_size": 51,
            "support": torch.linspace(0.0, 200.0, 51).to("cpu"),
            "batch_first": True,
            "accelerator": "cpu",
            "memory_systems": ["episodic", "semantic", "short"],
            "human_embedding_on_object_location": False,
        }

        lstm = LSTM(**config)
        lstm.forward(
            [
                "[{'human': 'Foo', 'object': 'laptop', 'object_location': 'desk', 'timestamp': 0}]",
                "[{'object': 'laptop', 'object_location': 'desk', 'num_generalized': 1}]",
                "[{'human': 'Bar', 'object': 'phone', 'object_location': 'lap', 'timestamp': 1}]",
            ]
        )
