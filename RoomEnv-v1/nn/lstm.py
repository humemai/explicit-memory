"""Deep Q-network architecture. Currently only LSTM is implemented."""
from copy import deepcopy

import numpy as np
import torch
from torch import nn


class LSTM(nn.Module):
    """A simple LSTM network."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        n_actions: int,
        embedding_dim: int,
        capacity: dict,
        entities: dict,
        include_human: str,
        batch_first: bool = True,
        human_embedding_on_object_location: bool = False,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        """Initialize the LSTM.

        Args
        ----
        hidden_size: hidden size of the LSTM
        num_layers: number of the LSTM layers
        n_actions: number of actions. This should be 3, at the moment.
        embedding_dim: entity embedding dimension (e.g., 32)
        capacity: the capacities of memory systems.
            e.g., {"episodic": 16, "semantic": 16, "short": 1}
        entities:
            e,g, {
            "humans": ["Foo", "Bar"],
            "objects": ["laptop", "phone"],
            "object_locations": ["desk", "lap"]}
        include_human:
            None: Don't include humans
            "sum": sum up the human embeddings with object / object_location embeddings.
            "cocnat": concatenate the human embeddings to object / object_location
                embeddings.
        batch_first: Should the batch dimension be the first or not.
        human_embedding_on_object_location: whether to superposition the human embedding
            on the tail (object location entity).
        device: "cpu" or "cuda"

        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.capacity = capacity
        self.entities = entities
        self.include_human = include_human
        self.memory_of_interest = list(self.capacity.keys())
        self.human_embedding_on_object_location = human_embedding_on_object_location
        self.device = device

        self.create_embeddings()
        if "episodic" in self.memory_of_interest:
            self.lstm_e = nn.LSTM(
                self.input_size_e,
                hidden_size,
                num_layers,
                batch_first=batch_first,
                device=self.device,
            )
            self.fc_e0 = nn.Linear(hidden_size, hidden_size, device=self.device)
            self.fc_e1 = nn.Linear(hidden_size, hidden_size, device=self.device)

        if "semantic" in self.memory_of_interest:
            self.lstm_s = nn.LSTM(
                self.input_size_s,
                hidden_size,
                num_layers,
                batch_first=batch_first,
                device=self.device,
            )
            self.fc_s0 = nn.Linear(hidden_size, hidden_size, device=self.device)
            self.fc_s1 = nn.Linear(hidden_size, hidden_size, device=self.device)

        if "short" in self.memory_of_interest:
            self.lstm_o = nn.LSTM(
                self.input_size_o,
                hidden_size,
                num_layers,
                batch_first=batch_first,
                device=self.device,
            )
            self.fc_o0 = nn.Linear(hidden_size, hidden_size, device=self.device)
            self.fc_o1 = nn.Linear(hidden_size, hidden_size, device=self.device)

        self.fc_final0 = nn.Linear(
            hidden_size * len(self.memory_of_interest),
            hidden_size * len(self.memory_of_interest),
            device=self.device,
        )
        self.fc_final1 = nn.Linear(
            hidden_size * len(self.memory_of_interest), n_actions, device=self.device
        )
        self.relu = nn.ReLU()

    def create_embeddings(self) -> None:
        """Create learnable embeddings."""
        self.word2idx = (
            ["<PAD>"]
            + self.entities["humans"]
            + self.entities["objects"]
            + self.entities["object_locations"]
        )
        self.word2idx = {word: idx for idx, word in enumerate(self.word2idx)}
        self.embeddings = nn.Embedding(
            len(self.word2idx), self.embedding_dim, device=self.device, padding_idx=0
        )
        self.input_size_s = self.embedding_dim * 2

        if (self.include_human is None) or (self.include_human.lower() == "sum"):
            self.input_size_e = self.embedding_dim * 2
            self.input_size_o = self.embedding_dim * 2

        elif self.include_human.lower() == "concat":
            self.input_size_e = self.embedding_dim * 3
            self.input_size_o = self.embedding_dim * 3
        else:
            raise ValueError(
                "include_human should be one of None, 'sum', or 'concat', "
                f"but {self.include_human} was given!"
            )

    def make_embedding(self, mem: dict, memory_type: str) -> torch.Tensor:
        """Create one embedding vector with summation and concatenation.

        Args
        ----
        mem: memory
            e.g, {"human": "Bob", "object": "laptop",
                  "object_location": "desk", "timestamp": 1}
        memory_type: "episodic", "semantic", or "short"

        Returns
        -------
        one embedding vector made from one memory element.

        """
        object_embedding = self.embeddings(
            torch.tensor(self.word2idx[mem["object"]], device=self.device)
        )
        object_location_embedding = self.embeddings(
            torch.tensor(self.word2idx[mem["object_location"]], device=self.device)
        )

        if memory_type.lower() == "semantic":
            final_embedding = torch.concat(
                [object_embedding, object_location_embedding]
            )

        elif memory_type.lower() in ["episodic", "short"]:
            human_embedding = self.embeddings(
                torch.tensor(self.word2idx[mem["human"]], device=self.device)
            )

            if self.include_human is None:
                final_embedding = torch.concat(
                    [object_embedding, object_location_embedding]
                )
            elif self.include_human.lower() == "sum":
                final_embedding = [object_embedding + human_embedding]

                if self.human_embedding_on_object_location:
                    final_embedding.append(object_location_embedding + human_embedding)
                else:
                    final_embedding.append(object_location_embedding)

                final_embedding = torch.concat(final_embedding)

            elif self.include_human.lower() == "concat":
                final_embedding = torch.concat(
                    [human_embedding, object_embedding, object_location_embedding]
                )
        else:
            raise ValueError

        return final_embedding

    def create_batch(self, x: list, memory_type: str) -> torch.Tensor:
        """Create one batch from data.

        Args
        ----
        x: a batch of episodic, semantic, or short memories.
        memory_type: "episodic", "semantic", or "short"

        Returns
        -------
        batch of embeddings.

        """

        if memory_type == "semantic":
            mem_pad = {
                "object": "<PAD>",
                "object_location": "<PAD>",
                "num_generalized": "<PAD>",
            }
        elif memory_type in ["episodic", "short"]:
            mem_pad = {
                "human": "<PAD>",
                "object": "<PAD>",
                "object_location": "<PAD>",
                "timestamp": "<PAD>",
            }
        else:
            raise ValueError

        mems_batch = deepcopy(x)
        for mems in mems_batch:
            for _ in range(self.capacity[memory_type] - len(mems)):
                # this is a dummy entry for padding.
                mems.append(mem_pad)

        batch_embeddings = []
        for mems in mems_batch:
            embeddings = []
            for mem in mems:
                mem_emb = self.make_embedding(mem, memory_type)
                embeddings.append(mem_emb)
            embeddings = torch.stack(embeddings)
            batch_embeddings.append(embeddings)

        batch_embeddings = torch.stack(batch_embeddings)
        # print(batch_embeddings.shape)

        return batch_embeddings

    def forward(self, x: np.ndarray) -> torch.Tensor:
        """Forward-pass.

        Args
        ----
        x is a batch of memories. Each element of the batch is a np.ndarray of dict
        memories.

        """
        to_concat = []
        if isinstance(x, dict):
            x = np.array([x])
        if "episodic" in self.memory_of_interest:
            batch_e = self.create_batch(
                [sample["episodic"] for sample in x],
                memory_type="episodic",
            )
            lstm_out_e, _ = self.lstm_e(batch_e)
            fc_out_e = self.relu(
                self.fc_e1(self.relu(self.fc_e0(lstm_out_e[:, -1, :])))
            )
            to_concat.append(fc_out_e)

        if "semantic" in self.memory_of_interest:
            batch_s = self.create_batch(
                [sample["semantic"] for sample in x],
                memory_type="semantic",
            )
            lstm_out_s, _ = self.lstm_s(batch_s)
            fc_out_s = self.relu(
                self.fc_s1(self.relu(self.fc_s0(lstm_out_s[:, -1, :])))
            )
            to_concat.append(fc_out_s)

        if "short" in self.memory_of_interest:
            batch_o = self.create_batch(
                [sample["short"] for sample in x],
                memory_type="short",
            )
            lstm_out_o, _ = self.lstm_o(batch_o)
            fc_out_o = self.relu(
                self.fc_o1(self.relu(self.fc_o0(lstm_out_o[:, -1, :])))
            )
            to_concat.append(fc_out_o)

        # dim=-1 is the feature dimension
        fc_out_all = torch.concat(to_concat, dim=-1)

        # fc_out has the dimension of (batch_size, 2)
        fc_out = self.fc_final1(self.relu(self.fc_final0(fc_out_all)))

        return fc_out
