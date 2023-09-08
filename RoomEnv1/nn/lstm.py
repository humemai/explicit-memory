"""Deep Q-network architecture. Currently only LSTM is implemented."""
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from explicit_memory.nn import NoisyLinear
from explicit_memory.utils import split_by_possessive


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
        atom_size: int,
        support: torch.Tensor,
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
        atom_size: parameter for NoisyNet, DuelingNet, and CategoricalDQN.
        support: parameter for NoisyNet, DuelingNet, and CategoricalDQN.
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
        self.atom_size = atom_size
        self.support = support
        self.n_actions = n_actions

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

        self.hidden_size_last = hidden_size * len(self.memory_of_interest)

        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(
            self.hidden_size_last, self.hidden_size_last, device=self.device
        )
        self.advantage_layer = NoisyLinear(
            self.hidden_size_last, self.n_actions * self.atom_size, device=self.device
        )

        # set value layer
        self.value_hidden_layer = NoisyLinear(
            self.hidden_size_last, self.hidden_size_last, device=self.device
        )
        self.value_layer = NoisyLinear(
            self.hidden_size_last, self.atom_size, device=self.device
        )

        self.relu = nn.ReLU()

    def dist(self, x: np.ndarray) -> torch.Tensor:
        """Get distribution for atoms."""
        to_concat = []
        if "episodic" in self.memory_of_interest:
            batch_e = [
                sample.item()["episodic"]
                if isinstance(sample, np.ndarray)
                else sample["episodic"]
                for sample in x
            ]
            batch_e = self.create_batch(batch_e, memory_type="episodic")
            lstm_out_e, _ = self.lstm_e(batch_e)
            fc_out_e = self.relu(
                self.fc_e1(self.relu(self.fc_e0(lstm_out_e[:, -1, :])))
            )
            to_concat.append(fc_out_e)

        if "semantic" in self.memory_of_interest:
            batch_s = [
                sample.item()["semantic"]
                if isinstance(sample, np.ndarray)
                else sample["semantic"]
                for sample in x
            ]
            batch_s = self.create_batch(batch_s, memory_type="semantic")
            lstm_out_s, _ = self.lstm_s(batch_s)
            fc_out_s = self.relu(
                self.fc_s1(self.relu(self.fc_s0(lstm_out_s[:, -1, :])))
            )
            to_concat.append(fc_out_s)

        if "short" in self.memory_of_interest:
            batch_o = [
                sample.item()["short"]
                if isinstance(sample, np.ndarray)
                else sample["short"]
                for sample in x
            ]
            batch_o = self.create_batch(batch_o, memory_type="short")
            lstm_out_o, _ = self.lstm_o(batch_o)
            fc_out_o = self.relu(
                self.fc_o1(self.relu(self.fc_o0(lstm_out_o[:, -1, :])))
            )
            to_concat.append(fc_out_o)
        # dim=-1 is the feature dimension
        feature = torch.concat(to_concat, dim=-1)

        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))

        advantage = self.advantage_layer(adv_hid).view(
            -1, self.n_actions, self.atom_size
        )
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans

        return dist

    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()

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

    def make_embedding(self, mem: list, memory_type: str) -> torch.Tensor:
        """Create one embedding vector with summation and concatenation.

        Args
        ----
        mem: memory as a quadruple: [head, relation, tail, num]
        memory_type: "episodic", "semantic", or "short"

        Returns
        -------
        one embedding vector made from one memory element.

        """
        if mem == ["<PAD>", "<PAD>", "<PAD>", "<PAD>"]:
            human, obj, obj_loc = "<PAD>", "<PAD>", "<PAD>"
        else:
            if memory_type == "semantic":
                obj = mem[0]
            else:
                human, obj = split_by_possessive(mem[0])

            obj_loc = mem[2]
        object_embedding = self.embeddings(
            torch.tensor(self.word2idx[obj], device=self.device)
        )
        object_location_embedding = self.embeddings(
            torch.tensor(self.word2idx[obj_loc], device=self.device)
        )

        if memory_type.lower() == "semantic":
            final_embedding = torch.concat(
                [object_embedding, object_location_embedding]
            )

        elif memory_type.lower() in ["episodic", "short"]:
            human_embedding = self.embeddings(
                torch.tensor(self.word2idx[human], device=self.device)
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
        mem_pad = ["<PAD>", "<PAD>", "<PAD>", "<PAD>"]

        # mems_batch = deepcopy(x)
        for mems in x:
            for _ in range(self.capacity[memory_type] - len(mems)):
                # this is a dummy entry for padding.
                mems.append(mem_pad)

        batch_embeddings = []
        for mems in x:
            embeddings = []
            for mem in mems:
                mem_emb = self.make_embedding(mem, memory_type)
                embeddings.append(mem_emb)
            embeddings = torch.stack(embeddings)
            batch_embeddings.append(embeddings)

        batch_embeddings = torch.stack(batch_embeddings)

        return batch_embeddings

    def forward(self, x: np.ndarray) -> torch.Tensor:
        """Forward-pass.

        Args
        ----
        x is a batch of memories. Each element of the batch is a np.ndarray of dict
        memories.

        """
        assert isinstance(x, np.ndarray)
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q
