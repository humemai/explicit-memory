"""DQN LSTM Baseline Agent for the RoomEnv2 environment."""

import datetime
import os
import random
import shutil
from copy import deepcopy
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from room_env.envs.room2 import RoomEnv2
from tqdm.auto import trange

from explicit_memory.utils import is_running_notebook, positional_encoding, write_yaml

from explicit_memory.utils.dqn import (
    target_hard_update,
    plot_results,
    save_final_results,
    save_validation,
    save_states_q_values_actions,
    select_action,
    update_model,
)


class LSTM(torch.nn.Module):
    """LSTM for the DQN baseline. This is different from the LSTM used for the memory-
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
        dueling_dqn: bool = True,
        fuse_information: Literal["concat", "sum"] = "sum",
        include_positional_encoding: bool = True,
        max_timesteps: int | None = None,
        max_strength: int | None = None,
    ) -> None:
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
        self.dueling_dqn = dueling_dqn
        self.fuse_information = fuse_information
        self.include_positional_encoding = include_positional_encoding
        self.max_timesteps = max_timesteps
        self.max_strength = max_strength

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

        if self.dueling_dqn:
            self.value_layer = torch.nn.Sequential(
                torch.nn.Linear(
                    self.hidden_size,
                    self.hidden_size,
                    device=self.device,
                ),
                torch.nn.ReLU(),
                torch.nn.Linear(
                    self.hidden_size,
                    1,
                    device=self.device,
                ),
            )

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
        """Create one embedding vector with summation or concatenation."""
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

        if self.dueling_dqn:
            value = self.value_layer(lstm_out)
            advantage = self.advantage_layer(lstm_out)
            q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            q = self.advantage_layer(lstm_out)

        return q


class History:
    def __init__(
        self,
        block_size: int = 6,
        action2str: dict = {0: "north", 1: "east", 2: "south", 3: "west", 4: "stay"},
    ) -> None:
        self.block_size = block_size
        self.blocks = [[]] * self.block_size

        self.action2str = action2str
        self.actions_str = list(self.action2str.values())
        self.actions_int = list(self.action2str.keys())

    def to_list(self) -> list:
        return [element for block in self.blocks for element in block]

    def add_block(self, block: list) -> None:
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


class DQNLSTMBaselineAgent:
    """DQN LSTM Baseline Agent interacting with environment.

    Based on https://github.com/Curt-Park/rainbow-is-all-you-need/
    """

    def __init__(
        self,
        env_str: str = "room_env:RoomEnv-v2",
        num_iterations: int = 10000,
        replay_buffer_size: int = 10000,
        warm_start: int = 1000,
        batch_size: int = 32,
        target_update_interval: int = 10,
        epsilon_decay_until: float = 10000,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.9,
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
        plotting_interval: int = 10,
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
        ddqn: bool = True,
        dueling_dqn: bool = True,
        default_root_dir: str = "./training_results/DQN/LSTM/baselines",
        run_handcrafted_baselines: bool = True,
    ) -> None:
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

        self.ddqn = ddqn
        self.dueling_dqn = dueling_dqn

        self.action2str = {0: "north", 1: "east", 2: "south", 3: "west", 4: "stay"}
        self.str2action = {v: k for k, v in self.action2str.items()}
        self.action_space = gym.spaces.Discrete(len(self.action2str))

        self.nn_params = nn_params
        self.nn_params["device"] = self.device
        self.nn_params["entities"] = self.env.unwrapped.entities
        self.nn_params["relations"] = self.env.unwrapped.relations
        self.nn_params["dueling_dqn"] = self.dueling_dqn
        self.nn_params["n_actions"] = len(self.action2str)
        self.nn_params["max_step_reward"] = self.env.unwrapped.num_questions_step

        self.val_filenames = []
        self.is_notebook = is_running_notebook()
        self.num_iterations = num_iterations
        self.plotting_interval = plotting_interval
        self.run_test = run_test

        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_until = epsilon_decay_until
        self.target_update_interval = target_update_interval
        self.gamma = gamma
        self.warm_start = warm_start
        assert self.batch_size <= self.warm_start <= self.replay_buffer_size

        # networks: dqn, dqn_target
        if self.nn_params["architecture"].lower() == "lstm":
            function_approximator = LSTM
            del self.nn_params["architecture"]
        elif self.nn_params["architecture"].lower() == "stare":
            raise NotImplementedError

        self.dqn = function_approximator(**self.nn_params)
        self.dqn_target = function_approximator(**self.nn_params)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.replay_buffer = ReplayBuffer(
            observation_type="dict", size=replay_buffer_size, batch_size=batch_size
        )

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        self.q_values = {"train": [], "val": [], "test": []}

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
        """Create the directory to store the results."""
        os.makedirs(self.default_root_dir, exist_ok=True)
        write_yaml(params_to_save, os.path.join(self.default_root_dir, "train.yaml"))

    def remove_results_from_disk(self) -> None:
        """Remove the results from the disk."""
        shutil.rmtree(self.default_root_dir)

    def fill_replay_buffer(self) -> None:
        """Make the replay buffer full in the beginning with the uniformly-sampled
        actions. The filling continues until it reaches the warm start size.

        """
        while len(self.replay_buffer) < self.warm_start:
            observations, info = self.env.reset()
            self.history = History(self.history_block_size)
            self.history.add_block(observations["room"])

            while True:
                actions_qa = self.history.answer_questions(observations["questions"])
                state = {"data": self.history.to_list()}
                # action_explore = self.str2action[self.history.explore("avoid_walls")]
                action_explore, q_values_ = select_action(
                    state=state,
                    greedy=False,
                    dqn=self.dqn,
                    epsilon=self.epsilon,
                    action_space=self.action_space,
                )
                action_pair = (actions_qa, self.action2str[action_explore])
                (
                    observations,
                    reward,
                    done,
                    truncated,
                    info,
                ) = self.env.step(action_pair)
                done = done or truncated

                self.history.add_block(observations["room"])

                next_state = {"data": self.history.to_list()}
                transition = [
                    deepcopy(state),
                    action_explore,
                    reward,
                    deepcopy(next_state),
                    done,
                ]
                self.replay_buffer.store(*transition)

                if done or len(self.replay_buffer) >= self.warm_start:
                    break

    def train(self) -> None:
        """Train the explore agent."""
        self.fill_replay_buffer()  # fill up the buffer till warm start size
        self.num_validation = 0

        self.epsilons = []
        self.training_loss = []
        self.scores = {"train": [], "val": [], "test": None}

        self.dqn.train()

        training_episode_begins = True

        score = 0
        bar = trange(1, self.num_iterations + 1)
        for self.iteration_idx in bar:
            if training_episode_begins:
                observations, info = self.env.reset()
                self.history = History(self.history_block_size)
                self.history.add_block(observations["room"])

            actions_qa = self.history.answer_questions(observations["questions"])
            state = {"data": self.history.to_list()}

            action_explore, q_values_ = select_action(
                state=state,
                greedy=False,
                dqn=self.dqn,
                epsilon=self.epsilon,
                action_space=self.action_space,
            )
            self.q_values["train"].append(q_values_)

            action_pair = (actions_qa, self.action2str[action_explore])
            (
                observations,
                reward,
                done,
                truncated,
                info,
            ) = self.env.step(action_pair)
            score += reward
            done = done or truncated

            if not done:
                self.history.add_block(observations["room"])
                next_state = {"data": self.history.to_list()}
                transition = [
                    deepcopy(state),
                    action_explore,
                    reward,
                    deepcopy(next_state),
                    done,
                ]
                self.replay_buffer.store(*transition)

                training_episode_begins = False

            else:  # if episode ends
                self.scores["train"].append(score)
                score = 0
                with torch.no_grad():
                    self.validate()

                training_episode_begins = True

            loss = update_model(
                replay_buffer=self.replay_buffer,
                optimizer=self.optimizer,
                device=self.device,
                dqn=self.dqn,
                dqn_target=self.dqn_target,
                ddqn=self.ddqn,
                gamma=self.gamma,
            )
            self.training_loss.append(loss)

            # linearly decrease epsilon
            self.epsilon = max(
                self.min_epsilon,
                self.epsilon
                - (self.max_epsilon - self.min_epsilon) / self.epsilon_decay_until,
            )
            self.epsilons.append(self.epsilon)

            # if hard update is needed
            if self.iteration_idx % self.target_update_interval == 0:
                target_hard_update(dqn=self.dqn, dqn_target=self.dqn_target)

            # plotting & show training results
            if (
                self.iteration_idx == self.num_iterations
                or self.iteration_idx % self.plotting_interval == 0
            ):
                self.plot_results("all", save_fig=True)

        with torch.no_grad():
            self.test()

        self.env.close()

    def validate_test_middle(self, val_or_test: str) -> tuple[list[float], dict]:
        scores_temp = []
        states = []
        q_values = []
        actions = []

        for idx in range(self.num_samples_for_results):
            if idx == self.num_samples_for_results - 1:
                save_results = True
            else:
                save_results = False
            score = 0

            observations, info = self.env.reset()
            self.history = History(self.history_block_size)
            self.history.add_block(observations["room"])

            while True:
                actions_qa = self.history.answer_questions(observations["questions"])
                state = {"data": self.history.to_list()}

                if save_results:
                    states.append(deepcopy(state))

                action_explore, q_values_ = select_action(
                    state=state,
                    greedy=True,
                    dqn=self.dqn,
                    epsilon=self.epsilon,
                    action_space=self.action_space,
                )
                if save_results:
                    q_values.append(q_values_)
                    actions.append(action_explore)
                    self.q_values[val_or_test].append(q_values_)

                action_pair = (actions_qa, self.action2str[action_explore])
                (
                    observations,
                    reward,
                    done,
                    truncated,
                    info,
                ) = self.env.step(action_pair)
                score += reward
                done = done or truncated

                if done:
                    break

                self.history.add_block(observations["room"])

            scores_temp.append(score)

        return scores_temp, states, q_values, actions

    def validate(self) -> None:
        self.dqn.eval()
        scores_temp, states, q_values, actions = self.validate_test_middle("val")

        save_validation(
            scores_temp=scores_temp,
            scores=self.scores,
            default_root_dir=self.default_root_dir,
            num_validation=self.num_validation,
            val_filenames=self.val_filenames,
            dqn=self.dqn,
        )
        save_states_q_values_actions(
            states, q_values, actions, self.default_root_dir, "val", self.num_validation
        )
        self.env.close()
        self.num_validation += 1
        self.dqn.train()

    def test(self, checkpoint: str = None) -> None:
        self.dqn.eval()
        self.env_config["seed"] = self.test_seed
        self.env = gym.make(self.env_str, **self.env_config)

        assert len(self.val_filenames) == 1
        self.dqn.load_state_dict(torch.load(self.val_filenames[0]))
        if checkpoint is not None:
            self.dqn.load_state_dict(torch.load(checkpoint))

        scores, states, q_values, actions = self.validate_test_middle("test")
        self.scores["test"] = scores

        save_final_results(
            self.scores, self.training_loss, self.default_root_dir, self.q_values, self
        )
        save_states_q_values_actions(
            states, q_values, actions, self.default_root_dir, "test"
        )

        self.plot_results("all", save_fig=True)
        self.env.close()
        self.dqn.train()

    def plot_results(self, to_plot: str = "all", save_fig: bool = False) -> None:
        """Plot things for DQN training.

        Args:
            to_plot: what to plot:
                training_td_loss
                epsilons
                training_score
                validation_score
                test_score
                q_values_train
                q_values_val
                q_values_test

        """
        plot_results(
            self.scores,
            self.training_loss,
            self.epsilons,
            self.q_values,
            self.iteration_idx,
            self.action_space.n.item(),
            self.num_iterations,
            self.env.unwrapped.total_maximum_episode_rewards,
            self.num_validation,
            self.num_samples_for_results,
            self.default_root_dir,
            to_plot,
            save_fig,
        )
