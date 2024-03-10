"""PPO LSTM Baseline Agent for the RoomEnv2 environment."""

import datetime
import os
import random
import shutil
from copy import deepcopy
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from room_env.envs.room2 import RoomEnv2
from tqdm.auto import tqdm
from torch.distributions import Categorical

from explicit_memory.utils import is_running_notebook, positional_encoding, write_yaml

from explicit_memory.utils.ppo import (
    plot_results,
    save_final_results,
    save_validation,
    select_action,
    update_model,
    save_states_actions_probs_values,
)


def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> None:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)


class LSTM(torch.nn.Module):
    """LSTM for the PPO baseline. This is different from the LSTM used for the memory-
    based agents."""

    def __init__(
        self,
        entities: list,
        relations: list,
        n_actions: int,
        max_step_reward: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        embedding_dim: int = 64,
        make_categorical_embeddings: bool = False,
        batch_first: bool = True,
        device: str = "cpu",
        fuse_information: Literal["concat", "sum"] = "sum",
        include_positional_encoding: bool = True,
        max_timesteps: int | None = None,
        max_strength: int | None = None,
        is_actor: bool = False,
        is_critic: bool = False,
    ) -> None:
        """Initialize the LSTM.

        Args:
            entities: List of entities.
            relations: List of relations.
            n_actions: Number of actions.
            max_step_reward: Maximum reward per step.
            hidden_size: Hidden size of the LSTM.
            num_layers: Number of layers in the LSTM.
            embedding_dim: Dimension of the embeddings.
            make_categorical_embeddings: Whether to make categorical embeddings.
            batch_first: Whether batch is first.
            device: Device to use.
            fuse_information: How to fuse information.
            include_positional_encoding: Whether to include positional encoding.
            max_timesteps: Maximum timesteps.
            max_strength: Maximum strength.
            is_actor: whether this is an actor or not.
            is_critic: whether this is a critic or not.

        """
        super().__init__()
        self.entities = entities
        self.relations = relations
        self.n_actions = n_actions
        self.max_step_reward = max_step_reward
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.make_categorical_embeddings = make_categorical_embeddings
        self.batch_first = batch_first
        self.device = device
        self.fuse_information = fuse_information
        self.include_positional_encoding = include_positional_encoding
        self.max_timesteps = max_timesteps
        self.max_strength = max_strength
        self.is_actor = is_actor
        self.is_critic = is_critic

        self.create_embeddings()
        if self.include_positional_encoding:
            assert self.max_timesteps is not None
            assert self.max_strength is not None
            self.positional_encoding = positional_encoding(
                positions=max(self.max_timesteps, self.max_strength) + 1,
                dimensions=self.input_size,
                scaling_factor=10000,
                return_tensor=True,
            )

        self.lstm = torch.nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            batch_first=self.batch_first,
            device=self.device,
        )

        if self.is_actor:
            self.advantage_layer = torch.nn.Sequential(
                torch.nn.Linear(
                    self.hidden_size,
                    self.hidden_size,
                    device=self.device,
                ),
                torch.nn.ReLU(),
                torch.nn.Linear(
                    self.hidden_size,
                    n_actions,
                    device=self.device,
                ),
            )
            init_layer_uniform(self.advantage_layer[-1], 3e-3)

        if self.is_critic:
            self.value_layer = nn.Sequential(
                nn.Linear(
                    self.hidden_size,
                    self.hidden_size,
                    device=self.device,
                ),
                nn.ReLU(),
                nn.Linear(
                    self.hidden_size,
                    1,
                    device=self.device,
                ),
            )
            init_layer_uniform(self.value_layer[-1], 3e-3)

        self.relu = torch.nn.ReLU()

    def create_embeddings(self) -> None:
        """Create learnable embeddings."""
        if isinstance(self.entities, dict):
            self.word2idx = (
                ["<PAD>"]
                + [name for names in self.entities.values() for name in names]
                + self.relations
            )
        elif isinstance(self.entities, list):
            self.word2idx = ["<PAD>"] + self.entities + self.relations
        else:
            raise ValueError(
                "entities should be either a list or a dictionary, but "
                f"{type(self.entities)} was given!"
            )
        self.word2idx = {word: idx for idx, word in enumerate(self.word2idx)}

        self.embeddings = torch.nn.Embedding(
            len(self.word2idx),
            self.embedding_dim,
            device=self.device,
            padding_idx=0,
        )

        if self.make_categorical_embeddings:
            # Assuming self.entities is a dictionary where keys are categories and
            # values are lists of entity names

            # Create a dictionary to keep track of starting index for each category
            category_start_indices = {}
            current_index = 1  # Start from 1 to skip the <PAD> token
            for category, names in self.entities.items():
                category_start_indices[category] = current_index
                current_index += len(names)

            # Re-initialize embeddings by category
            for category, start_idx in category_start_indices.items():
                end_idx = start_idx + len(self.entities[category])
                init_vector = torch.randn(self.embedding_dim, device=self.device)
                self.embeddings.weight.data[start_idx:end_idx] = init_vector.repeat(
                    end_idx - start_idx, 1
                )
            # Note: Relations are not re-initialized by category, assuming they are
            # separate from entities

        if self.fuse_information == "concat":
            self.input_size = self.embedding_dim * 3
        elif self.fuse_information == "sum":
            self.input_size = self.embedding_dim
        else:
            raise ValueError(
                f"fuse_information should be one of 'concat' or 'sum', but "
                f"{self.fuse_information} was given!"
            )

    def make_embedding(self, obs: list[str]) -> torch.Tensor:
        """Create one embedding vector with summation or concatenation.

        Args:
            obs: Observation.

        Returns:
            final_embedding: Final embedding.

        """
        if obs == ["<PAD>", "<PAD>", "<PAD>", "<PAD>"]:
            if self.fuse_information == "sum":
                return self.embeddings(
                    torch.tensor(self.word2idx["<PAD>"], device=self.device)
                )
            else:
                final_embedding = torch.concat(
                    [
                        self.embeddings(
                            torch.tensor(self.word2idx["<PAD>"], device=self.device)
                        ),
                        self.embeddings(
                            torch.tensor(self.word2idx["<PAD>"], device=self.device)
                        ),
                        self.embeddings(
                            torch.tensor(self.word2idx["<PAD>"], device=self.device)
                        ),
                    ]
                )
                return final_embedding

        head_embedding = self.embeddings(
            torch.tensor(self.word2idx[obs[0]], device=self.device)
        )
        relation_embedding = self.embeddings(
            torch.tensor(self.word2idx[obs[1]], device=self.device)
        )
        tail_embedding = self.embeddings(
            torch.tensor(self.word2idx[obs[2]], device=self.device)
        )
        if self.fuse_information == "concat":
            final_embedding = torch.concat(
                [head_embedding, relation_embedding, tail_embedding]
            )
        elif self.fuse_information == "sum":
            final_embedding = head_embedding + relation_embedding + tail_embedding
        else:
            raise ValueError(
                f"fuse_information should be one of 'concat' or 'sum', but "
                f"{self.fuse_information} was given!"
            )

        if self.include_positional_encoding:
            final_embedding += self.positional_encoding[obs[3]]

        return final_embedding

    def forward(self, x: np.ndarray) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input.

        Returns:
            (action, distribution), or value

        """
        obs_pad = ["<PAD>", "<PAD>", "<PAD>", "<PAD>"]
        batch = [sample["data"] for sample in x]
        max_len = 0
        for sample in batch:
            if len(sample) > max_len:
                max_len = len(sample)

        for observations in batch:
            while len(observations) < max_len:
                observations.append(obs_pad)

        batch_embeddings = []
        for observations in batch:
            embeddings = []
            for obs in observations:
                obs_emb = self.make_embedding(obs)
                embeddings.append(obs_emb)
            embeddings = torch.stack(embeddings)
            batch_embeddings.append(embeddings)

        batch_embeddings = torch.stack(batch_embeddings)

        lstm_out, _ = self.lstm(batch_embeddings)
        lstm_out = lstm_out[:, -1, :]

        if self.is_actor:
            logits = self.advantage_layer(lstm_out)
            dist = Categorical(logits=logits)
            action = dist.sample()

            return action, dist

        if self.is_critic:
            value = self.value_layer(lstm_out)

            return value


class History:
    def __init__(
        self,
        block_size: int = 6,
        action2str: dict = {0: "north", 1: "east", 2: "south", 3: "west", 4: "stay"},
    ) -> None:
        """Initialize the history.

        Args:
            block_size: Block size.
            action2str: Action number to action string.

        """
        self.block_size = block_size
        self.blocks = [[]] * self.block_size

        self.action2str = action2str
        self.actions_str = list(self.action2str.values())
        self.actions_int = list(self.action2str.keys())

    def to_list(self) -> list:
        """Convert the history to a list."""
        return [element for block in self.blocks for element in block]

    def add_block(self, block: list) -> None:
        """Add a block to the history."""
        self.blocks = self.blocks[1:] + [block]

    def __repr__(self) -> str:
        return str(self.blocks)

    def answer_question(self, question: list) -> str:
        """Answer a question, by going through the history in backwards."""
        assert len(question) == 4 and question[2] == "?"
        for obs in self.to_list()[::-1]:
            if obs[0] == question[0] and obs[1] == question[1]:
                return obs[2]

    def answer_questions(self, questions: list[list[str]]) -> list[str]:
        """Answer a list of questions."""
        return [self.answer_question(question) for question in questions]

    def find_agent_current_location(self) -> str | None:
        """Find the current location of the agent, by going through the history in
        backwards."""
        for obs in self.to_list()[::-1]:
            if obs[0] == "agent" and obs[1] == "atlocation":
                return obs[2]

        return None

    def explore(self, explore_policy: str) -> str:
        """Explore the environment. The agent can either explore randomly or avoid
        walls. This is not reinforcement learning, but a simple heuristic.

        Args:
            explore_policy: Explore policy.

        Returns:
            action_explore: Action to explore.

        """
        if explore_policy not in ["random", "avoid_walls"]:
            raise ValueError(f"Unknown explore policy: {explore_policy}")

        if explore_policy == "random":
            action_explore = random.choice(self.actions_str)
        elif explore_policy == "avoid_walls":
            current_agent_location = self.find_agent_current_location()
            if current_agent_location is None:
                action_explore = random.choice(self.actions_str)
            else:
                relevant_observations = []
                relevant_observations += [
                    obs
                    for obs in self.to_list()[::-1]
                    if obs[0] == current_agent_location
                    and obs[1] in ["north", "east", "south", "west"]
                ]
                # we know the agent's current location but there is no memory about map
                if len(relevant_observations) == 0:
                    action_explore = random.choice(self.actions_str)
                else:
                    # we know the agent's current location and there is at least one
                    # memory about the map and we want to avoid the walls

                    to_take = []
                    to_avoid = []

                    for obs in relevant_observations:
                        if isinstance(obs, dict):
                            continue
                        if obs[2].split("_")[0] == "room":
                            to_take.append(obs[1])
                        elif obs[2] == "wall":
                            if obs[1] not in to_avoid:
                                to_avoid.append(obs[1])

                    if len(to_take) > 0:
                        action_explore = random.choice(to_take)
                    else:
                        options = deepcopy(self.actions_str)
                        for e in to_avoid:
                            options.remove(e)

                        action_explore = random.choice(options)

        return action_explore


class PPOLSTMBaselineAgent:
    """PPO LSTM Baseline Agent interacting with environment.

    Based on https://github.com/Curt-Park/rainbow-is-all-you-need/
    """

    def __init__(
        self,
        env_str: str = "room_env:RoomEnv-v2",
        num_episodes: int = 10,
        num_rollouts: int = 2,
        epoch_per_rollout: int = 64,
        batch_size: int = 128,
        gamma: float = 0.9,
        lam: float = 0.8,
        epsilon: float = 0.2,
        entropy_weight: float = 0.005,
        history_block_size: int = 6,
        nn_params: dict = {
            "architecture": "lstm",
            "hidden_size": 64,
            "num_layers": 2,
            "embedding_dim": 64,
            "fuse_information": "sum",
            "include_positional_encoding": True,
            "max_timesteps": 100,
            "max_strength": 100,
        },
        run_test: bool = True,
        num_samples_for_results: int = 10,
        train_seed: int = 5,
        test_seed: int = 0,
        device: str = "cpu",
        env_config: dict = {
            "question_prob": 1.0,
            "terminates_at": 99,
            "randomize_observations": "objects",
            "room_size": "l",
            "rewards": {"correct": 1, "wrong": 0, "partial": 0},
            "make_everything_static": False,
            "num_total_questions": 1000,
            "question_interval": 1,
            "include_walls_in_observations": True,
        },
        default_root_dir: str = "./training_results/PPO/baselines/",
        run_handcrafted_baselines: bool = False,
    ) -> None:
        """Initialize the PPO LSTM Baseline Agent.

        Args:
            env_str: Environment string.
            num_episodes: Number of episodes.
            num_rollouts: Number of rollouts.
            epoch_per_rollout: Epochs per rollout.
            batch_size: Batch size.
            gamma: Gamma. Discount factor.
            lam: GAE lambda parameter.
            epsilon: Epsilon. Clipping parameter.
            entropy_weight: Entropy weight.
            history_block_size: The number of blocks for history
            nn_params: Neural network parameters.
            run_test: Whether to run test.
            num_samples_for_results: Number of samples for results.
            train_seed: Train seed.
            test_seed: Test seed.
            device: Device. "cpu" or "cuda".
            env_config: Environment configuration.
            default_root_dir: Default root directory.
            run_handcrafted_baselines: Whether to run handcrafted baselines.

        """
        params_to_save = deepcopy(locals())
        del params_to_save["self"]

        self.history_block_size = history_block_size

        self.train_seed = train_seed
        self.test_seed = test_seed
        env_config["seed"] = self.train_seed

        self.env_str = env_str
        self.env_config = env_config
        self.num_samples_for_results = num_samples_for_results
        self.env = gym.make(self.env_str, **self.env_config)

        self.default_root_dir = os.path.join(
            default_root_dir, str(datetime.datetime.now())
        )
        self._create_directory(params_to_save)
        self.run_handcrafted_baselines = run_handcrafted_baselines

        self.device = torch.device(device)
        print(f"Running on {self.device}")

        self.action2str = {0: "north", 1: "east", 2: "south", 3: "west", 4: "stay"}
        self.str2action = {v: k for k, v in self.action2str.items()}
        self.action_space = gym.spaces.Discrete(len(self.action2str))

        self.nn_params = nn_params
        self.nn_params["device"] = self.device
        self.nn_params["entities"] = self.env.unwrapped.entities
        self.nn_params["relations"] = self.env.unwrapped.relations
        self.nn_params["n_actions"] = len(self.action2str)
        self.nn_params["max_step_reward"] = self.env.unwrapped.num_questions_step

        self.val_filenames = []
        self.is_notebook = is_running_notebook()

        self.num_episodes = num_episodes
        self.num_rollouts = num_rollouts
        self.epoch_per_rollout = epoch_per_rollout
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.entropy_weight = entropy_weight

        self.run_test = run_test

        self.num_steps_in_episode = self.env.unwrapped.terminates_at + 1
        self.total_maximum_episode_rewards = (
            self.env.unwrapped.total_maximum_episode_rewards
        )

        assert (self.num_rollouts % self.num_episodes) == 0 or (
            self.num_episodes % self.num_rollouts
        ) == 0

        self.num_steps_per_rollout = int(
            self.num_episodes / self.num_rollouts * self.num_steps_in_episode
        )

        # networks: dqn, dqn_target
        if self.nn_params["architecture"].lower() == "lstm":
            function_approximator = LSTM
            del self.nn_params["architecture"]
        elif self.nn_params["architecture"].lower() == "stare":
            raise NotImplementedError

        self.actor = function_approximator(
            **self.nn_params, is_actor=True, is_critic=False
        )
        self.critic = function_approximator(
            **self.nn_params, is_actor=False, is_critic=True
        )

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters())
        self.critic_optimizer = optim.Adam(self.critic.parameters())

        # global stats to save
        self.actor_losses, self.critic_losses = [], []  # training loss
        self.states_all = {"train": [], "val": [], "test": []}
        self.scores_all = {"train": [], "val": [], "test": None}
        self.actions_all = {"train": [], "val": [], "test": []}
        self.actor_probs_all = {"train": [], "val": [], "test": []}
        self.critic_values_all = {"train": [], "val": [], "test": []}

        if self.run_handcrafted_baselines:
            self._run_explore_baselines()

    def _run_explore_baselines(self) -> None:
        """Run the explore baselines."""
        env = RoomEnv2(**self.env_config)
        observations, info = env.reset()
        env.render("image", save_fig_dir=self.default_root_dir)

        del env

        env_config = deepcopy(self.env_config)

        results = {}
        for explore_policy in ["random", "avoid_walls"]:
            results[str((explore_policy))] = []
            scores = []
            for test_seed in [0, 1, 2, 3, 4]:
                score = 0
                env_config["seed"] = test_seed
                env = gym.make(self.env_str, **env_config)
                observations, info = env.reset()
                self.history = History(self.history_block_size, self.action2str)
                self.history.add_block(observations["room"])

                while True:
                    actions_qa = self.history.answer_questions(
                        observations["questions"]
                    )
                    action_explore = self.history.explore(explore_policy)

                    action_pair = (actions_qa, action_explore)
                    (
                        observations,
                        reward,
                        done,
                        truncated,
                        info,
                    ) = env.step(action_pair)
                    score += reward
                    done = done or truncated

                    self.history.add_block(observations["room"])

                    if done:
                        break

                scores.append(score)

            results[str((explore_policy))] = {
                "mean": np.mean(scores).item(),
                "std": np.std(scores).item(),
            }

        write_yaml(results, os.path.join(self.default_root_dir, "handcrafted.yaml"))

    def _create_directory(self, params_to_save: dict) -> None:
        """Create the directory to store the results.

        Args:
            params_to_save: (hyper) parameters to save.

        """
        os.makedirs(self.default_root_dir, exist_ok=True)
        write_yaml(params_to_save, os.path.join(self.default_root_dir, "train.yaml"))

    def remove_results_from_disk(self) -> None:
        """Remove the results from the disk."""
        shutil.rmtree(self.default_root_dir)

    def create_empty_rollout_buffer(self) -> tuple[list, list, list, list, list, list]:
        """Create empty buffer for training.

        Make sure to call this before each rollout.

        Returns:
            states_buffer: The states.
            actions_buffer: The actions.
            rewards_buffer: The rewards.
            values_buffer: The values.
            masks_buffer: The masks.
            log_probs_buffer: The log probabilities.

        """
        # memory for training
        states_buffer: list[dict] = []  # this has to be a list of dictionaries
        actions_buffer: list[torch.Tensor] = []
        rewards_buffer: list[torch.Tensor] = []
        values_buffer: list[torch.Tensor] = []
        masks_buffer: list[torch.Tensor] = []
        log_probs_buffer: list[torch.Tensor] = []

        return (
            states_buffer,
            actions_buffer,
            rewards_buffer,
            values_buffer,
            masks_buffer,
            log_probs_buffer,
        )

    def step(
        self,
        observations: dict,
        is_train_val_test: str,
        states_buffer: list | None = None,
        actions_buffer: list | None = None,
        values_buffer: list | None = None,
        log_probs_buffer: list | None = None,
        append_states_actions_probs_values: bool = False,
        append_states: bool = False,
    ) -> tuple[int, bool, dict]:
        """Take a step in the environment.

        Args:
            observations: observations
            is_train_val_test: "train", "val", or "test"
            states_buffer: states buffer
            actions_buffer: actions buffer
            values_buffer: values buffer
            log_probs_buffer: log probs buffer
            append_states_actions_probs_values: whether to append states, actions,
                probs, and values, to save them later
            append_states: whether to append states, to save them later

        """
        self.history.add_block(observations["room"])
        state = {"data": self.history.to_list()}

        action_explore, actor_probs, critic_value = select_action(
            actor=self.actor,
            critic=self.critic,
            state=state,
            is_test=(is_train_val_test in ["val", "test"]),
            states=states_buffer,
            actions=actions_buffer,
            values=values_buffer,
            log_probs=log_probs_buffer,
        )

        if append_states_actions_probs_values:
            if append_states:
                # state is a list, which is a mutable object. So, we need to
                # deepcopy it.
                self.states_all[is_train_val_test].append(deepcopy(state))
            else:
                self.states_all[is_train_val_test].append(None)

            self.actions_all[is_train_val_test].append(action_explore)
            self.actor_probs_all[is_train_val_test].append(actor_probs)
            self.critic_values_all[is_train_val_test].append(critic_value)

        actions_qa = self.history.answer_questions(observations["questions"])
        action_pair = (actions_qa, self.action2str[action_explore])
        (
            observations,
            reward,
            done,
            truncated,
            info,
        ) = self.env.step(action_pair)
        done = done or truncated

        return reward, done, observations

    def train(self) -> None:
        """Train the agent."""

        self.num_validation = 0
        new_episode_starts = True
        score = 0

        self.actor.train()
        self.critic.train()

        for _ in tqdm(range(self.num_rollouts)):
            (
                states_buffer,
                actions_buffer,
                rewards_buffer,
                values_buffer,
                masks_buffer,
                log_probs_buffer,
            ) = self.create_empty_rollout_buffer()

            for _ in range(self.num_steps_per_rollout):
                if new_episode_starts:
                    observations, info = self.env.reset()
                    self.history = History(self.history_block_size)

                reward, done, observations = self.step(
                    observations=observations,
                    is_train_val_test="train",
                    states_buffer=states_buffer,
                    actions_buffer=actions_buffer,
                    values_buffer=values_buffer,
                    log_probs_buffer=log_probs_buffer,
                    append_states_actions_probs_values=True,
                    append_states=False,
                )
                score += reward

                reward = np.reshape(reward, (1, -1)).astype(np.float64)
                rewards_buffer.append(torch.FloatTensor(reward).to(self.device))

                done_ = np.reshape(done, (1, -1))
                masks_buffer.append(torch.FloatTensor(1 - done_).to(self.device))

                # if episode ends
                if done:
                    self.scores_all["train"].append(score)
                    with torch.no_grad():
                        self.validate()

                    score = 0
                    new_episode_starts = True

                else:
                    new_episode_starts = False

            # this block is important. We have to get the next_state
            history_original = deepcopy(self.history)
            self.history.add_block(observations["room"])
            next_state = {"data": self.history.to_list()}

            # we have to reset the history to the original one after next_state is
            # computed
            self.history = history_original

            actor_loss, critic_loss = update_model(
                next_state,
                states_buffer,
                actions_buffer,
                rewards_buffer,
                values_buffer,
                masks_buffer,
                log_probs_buffer,
                self.gamma,
                self.lam,
                self.epoch_per_rollout,
                self.batch_size,
                self.epsilon,
                self.entropy_weight,
                self.actor,
                self.critic,
                self.actor_optimizer,
                self.critic_optimizer,
            )

            self.actor_losses.append(actor_loss)
            self.critic_losses.append(critic_loss)

            self.plot_results("all", True)

        with torch.no_grad():
            self.test()

        self.env.close()
        save_states_actions_probs_values(
            self.states_all["train"],
            self.actions_all["train"],
            self.actor_probs_all["train"],
            self.critic_values_all["train"],
            self.default_root_dir,
            "train",
        )

    def validate_test_middle(self, val_or_test: str) -> list[float]:
        """A function shared by validation and test in the middle.

        Args:
            val_or_test: "val" or "test"


        Returns:
            scores: Episode rewards. The number of elements is num_samples_for_results.

        """
        scores = []

        for idx in range(self.num_samples_for_results):
            if idx == self.num_samples_for_results - 1:
                save_results = True
            else:
                save_results = False

            score = 0

            observations, info = self.env.reset()
            self.history = History(self.history_block_size)

            while True:
                reward, done, observations = self.step(
                    observations=observations,
                    is_train_val_test=val_or_test,
                    states_buffer=None,
                    actions_buffer=None,
                    values_buffer=None,
                    log_probs_buffer=None,
                    append_states_actions_probs_values=save_results,
                    append_states=save_results,
                )

                score += reward

                if done:
                    break

            scores.append(score)

        return scores

    def validate(self) -> None:
        """Validate the agent."""
        self.actor.eval()
        self.critic.eval()

        scores = self.validate_test_middle("val")

        save_validation(
            scores=scores,
            scores_all_val=self.scores_all["val"],
            default_root_dir=self.default_root_dir,
            num_validation=self.num_validation,
            val_filenames=self.val_filenames,
            actor=self.actor,
            critic=self.critic,
        )

        start = self.num_validation * self.num_steps_in_episode
        end = (self.num_validation + 1) * self.num_steps_in_episode

        save_states_actions_probs_values(
            self.states_all["val"][start:end],
            self.actions_all["val"][start:end],
            self.actor_probs_all["val"][start:end],
            self.critic_values_all["val"][start:end],
            self.default_root_dir,
            "val",
            self.num_validation,
        )

        self.env.close()
        self.num_validation += 1
        self.actor.train()
        self.critic.train()

    def test(self, checkpoint: str = None) -> None:
        self.env_config["seed"] = self.test_seed
        self.env = gym.make(self.env_str, **self.env_config)
        self.actor.eval()
        self.critic.eval()

        assert len(self.val_filenames) == 1
        self.actor.load_state_dict(
            torch.load(os.path.join(self.val_filenames[0], "actor.pt"))
        )
        self.critic.load_state_dict(
            torch.load(os.path.join(self.val_filenames[0], "critic.pt"))
        )
        if checkpoint is not None:
            self.actor.load_state_dict(os.path.join(torch.load(checkpoint), "actor.pt"))
            self.critic.load_state_dict(
                os.path.join(torch.load(checkpoint), "critic.pt")
            )

        scores = self.validate_test_middle("test")

        self.scores_all["test"] = scores

        save_states_actions_probs_values(
            self.states_all["test"],
            self.actions_all["test"],
            self.actor_probs_all["test"],
            self.critic_values_all["test"],
            self.default_root_dir,
            "test",
        )

        save_final_results(
            self.scores_all,
            self.actor_losses,
            self.critic_losses,
            self.default_root_dir,
            self,
        )

        self.plot_results("all", True)
        self.env.close()
        self.actor.train()
        self.critic.train()

    def plot_results(self, to_plot: str = "all", save_fig: bool = False) -> None:
        """Plot things for ppo training.

        Args:
            to_plot: what to plot:
                all: everything
                actor_loss: actor loss
                critic_loss: critic loss
                scores: train, val, and test scores
                actor_probs_train: actor probabilities for training
                actor_probs_val: actor probabilities for validation
                actor_probs_test: actor probabilities for test
                critic_values_train: critic values for training
                critic_values_val: critic values for validation
                critic_values_test: critic values for test

        """
        plot_results(
            self.scores_all,
            self.actor_losses,
            self.critic_losses,
            self.actor_probs_all,
            self.critic_values_all,
            self.num_validation,
            self.action_space.n.item(),
            self.num_episodes,
            self.total_maximum_episode_rewards,
            self.default_root_dir,
            to_plot,
            save_fig,
        )
