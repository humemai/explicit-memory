import unittest

from model import LSTM


class LSTMTest(unittest.TestCase):
    def test_all(self) -> None:
        configs = []

        for hidden_size in [16, 32, 64]:
            for num_layers in [1, 2, 4]:
                for num_actions in [2, 3]:
                    for embedding_dim in [32, 64]:
                        for capacity in [4, 8, 16, 32, 64]:
                            for include_human in [None, "sum", "concat"]:
                                for batch_first in [True, False]:
                                    for memory_systems in [
                                        ["episodic"],
                                        ["semantic"],
                                        ["episodic", "semantic"],
                                        ["episodic", "semantic", "short"],
                                    ]:
                                        for human_embedding_on_object_location in [
                                            True,
                                            False,
                                        ]:
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
                                                    "entities": {
                                                        "humans": ["Foo", "Bar"],
                                                        "objects": ["laptop", "phone"],
                                                        "object_locations": [
                                                            "desk",
                                                            "lap",
                                                        ],
                                                    },
                                                    "include_human": include_human,
                                                    "batch_first": batch_first,
                                                    "accelerator": "cpu",
                                                    "memory_systems": memory_systems,
                                                    "human_embedding_on_object_location": human_embedding_on_object_location,
                                                }
                                            )
        for config in configs:
            lstm = LSTM(**config)

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
                "humans": ["Foo", "Bar"],
                "objects": ["laptop", "phone"],
                "object_locations": [
                    "desk",
                    "lap",
                ],
            },
            "include_human": "sum",
        }
        lstm = LSTM(**config)
        lstm.forward(
            [
                "[{'human': 'Foo', 'object': 'laptop', 'object_location': 'desk', 'timestamp': 0}]",
                "[{'object': 'laptop', 'object_location': 'desk', 'num_generalized': 1}]",
                "[{'human': 'Bar', 'object': 'phone', 'object_location': 'lap', 'timestamp': 1}]",
            ]
        )
