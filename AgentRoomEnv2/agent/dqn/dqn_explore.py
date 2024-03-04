"""DQN explore Agent for the RoomEnv2 environment."""

import os
from copy import deepcopy
import gymnasium as gym
import numpy as np
import torch
from tqdm.auto import trange

from explicit_memory.policy import (
    answer_question,
    encode_observation,
    explore,
    manage_memory,
)
from explicit_memory.utils import write_yaml, read_pickle, read_yaml

from explicit_memory.utils.dqn import (
    target_hard_update,
    select_action,
    update_model,
)

from .dqn import DQNAgent


class DQNExploreAgent(DQNAgent):
    """DQN explore Agent interacting with environment.

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
        capacity: dict = {
            "episodic": 16,
            "episodic_agent": 0,
            "semantic": 16,
            "semantic_map": 0,
            "short": 1,
        },
        pretrain_semantic: bool = None,
        nn_params: dict = {
            "architecture": "lstm",
            "hidden_size": 64,
            "num_layers": 2,
            "embedding_dim": 64,
            "make_categorical_embeddings": False,
            "v1_params": None,
            "v2_params": {},
            "memory_of_interest": [
                "episodic",
                "semantic",
            ],
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
        mm_policy: str = "neural",
        mm_agent_path: (
            str | None
        ) = "trained-agents/lstm-mm/2023-12-28 18:13:03.001952/agent.pkl",
        qa_policy: str = "episodic_semantic",
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
        default_root_dir: str = "./training_results/DQN/LSTM/explore",
        run_handcrafted_baselines: dict | None = [
            {
                "mm": mm,
                "qa": qa,
                "explore": explore,
                "pretrain_semantic": pretrain_semantic,
            }
            for mm in ["random", "episodic", "semantic"]
            for qa in ["episodic_semantic"]
            for explore in ["random", "avoid_walls"]
            for pretrain_semantic in [False, "exclude_walls"]
        ],
    ) -> None:
        """Initialization.

        Args:
            env_str: This has to be "room_env:RoomEnv-v2"
            num_iterations: The number of iterations to train the agent.
            replay_buffer_size: The size of the replay buffer.
            warm_start: The number of samples to fill the replay buffer with, before
                starting
            batch_size: The batch size for training This is the amount of samples sampled
                from the replay buffer.
            target_update_interval: The rate to update the target network.
            epsilon_decay_until: The iteration index until which to decay epsilon.
            max_epsilon: The maximum epsilon.
            min_epsilon: The minimum epsilon.
            gamma: The discount factor.
            capacity: The capacity of each human-like memory systems.
            pretrain_semantic: Whether or not to pretrain the semantic memory system.
            nn_params: The parameters for the DQN (function approximator).
            run_test: Whether or not to run test.
            num_samples_for_results: The number of samples to validate / test the agent.
            plotting_interval: The interval to plot the results.
            train_seed: The random seed for train.
            test_seed: The random seed for test.
            device: The device to run the agent on. This is either "cpu" or "cuda".
            mm_policy: Memory management policy. Choose one of "generalize",
                "random", "rl", or "neural"
            mm_agent_path: The path to the memory management agent. This is only used
                when mm_policy == "rl".
            qa_policy: question answering policy Choose one of "episodic_semantic",
                "random", or "neural". qa_policy shouldn't be trained with RL. There is no
                sequence of states / actions to learn from.
            env_config: The configuration of the environment.
                question_prob: The probability of a question being asked at every
                    observation.
                terminates_at: The maximum number of steps to take in an episode.
                seed: seed for env
                room_size: The room configuration to use. Choose one of "dev", "xxs", "xs",
                    "s", "m", or "l".
            ddqn: wehther to use double dqn
            dueling_dqn: whether to use dueling dqn
            default_root_dir: default root directory to store the results.
            run_handcrafted_baselines: Whether or not to run handcrafted baselines.

        """
        all_params = deepcopy(locals())
        del all_params["self"]
        del all_params["__class__"]
        self.all_params = deepcopy(all_params)
        del all_params["mm_agent_path"]

        self.action2str = {0: "north", 1: "east", 2: "south", 3: "west", 4: "stay"}
        self.action_space = gym.spaces.Discrete(len(self.action2str))

        all_params["nn_params"]["n_actions"] = len(self.action2str)
        all_params["explore_policy"] = "rl"
        super().__init__(**all_params)
        write_yaml(self.all_params, os.path.join(self.default_root_dir, "train.yaml"))

        if self.mm_policy == "neural":
            self.mm_agent = read_pickle(mm_agent_path)
            self.mm_agent.dqn.eval()
            self.mm_policy_model = self.mm_agent.dqn
        else:
            self.mm_policy_model = None

        with torch.no_grad():
            test_mean, test_std = self.run_neural_baseline()

        handcrafted = read_yaml(os.path.join(self.default_root_dir, "handcrafted.yaml"))
        handcrafted[
            "{"
            "mm"
            ": "
            "neural"
            ", "
            "qa"
            ": "
            "episodic_semantic"
            ", "
            "explore"
            ": "
            "avoid_walls"
            "}"
        ] = {"mean": test_mean, "std": test_std}
        write_yaml(handcrafted, os.path.join(self.default_root_dir, "handcrafted.yaml"))
        self.env_config["seed"] = self.train_seed
        self.env = gym.make(self.env_str, **self.env_config)

    def run_neural_baseline(self) -> None:
        """Run the neural baseline."""
        self.env_config["seed"] = self.test_seed
        self.env = gym.make(self.env_str, **self.env_config)
        scores = []

        for _ in range(self.num_samples_for_results):
            score = 0

            self.init_memory_systems()
            observations, info = self.env.reset()

            observations["room"] = self.manage_agent_and_map_memory(
                observations["room"]
            )

            for obs in observations["room"]:
                encode_observation(self.memory_systems, obs)
                manage_memory(
                    self.memory_systems,
                    self.mm_policy,
                    self.mm_policy_model,
                    split_possessive=False,
                )

            while True:
                actions_qa = [
                    answer_question(
                        self.memory_systems,
                        self.qa_policy,
                        question,
                        split_possessive=False,
                    )
                    for question in observations["questions"]
                ]
                action_explore = explore(self.memory_systems, "avoid_walls")

                action_pair = (actions_qa, action_explore)
                (
                    observations,
                    reward,
                    done,
                    truncated,
                    info,
                ) = self.env.step(action_pair)
                score += reward
                done = done or truncated

                observations["room"] = self.manage_agent_and_map_memory(
                    observations["room"]
                )

                for obs in observations["room"]:
                    encode_observation(self.memory_systems, obs)
                    manage_memory(
                        self.memory_systems,
                        self.mm_policy,
                        self.mm_policy_model,
                        split_possessive=False,
                    )

                if done:
                    break

            scores.append(score)

        return np.mean(scores).item(), np.std(scores).item()

    def fill_replay_buffer(self) -> None:
        """Make the replay buffer full in the beginning with the uniformly-sampled
        actions. The filling continues until it reaches the warm start size.

        For explore_policy == "rl"

        """
        while len(self.replay_buffer) < self.warm_start:
            self.init_memory_systems()
            observations, info = self.env.reset()

            observations["room"] = self.manage_agent_and_map_memory(
                observations["room"]
            )

            for obs in observations["room"]:
                encode_observation(self.memory_systems, obs)
                manage_memory(
                    self.memory_systems,
                    self.mm_policy,
                    self.mm_policy_model,
                    split_possessive=False,
                )

            while True:
                actions_qa = [
                    answer_question(
                        self.memory_systems,
                        self.qa_policy,
                        question,
                        split_possessive=False,
                    )
                    for question in observations["questions"]
                ]
                state = self.memory_systems.return_as_a_dict_list()
                action, q_values_ = select_action(
                    state=state,
                    greedy=False,
                    dqn=self.dqn,
                    epsilon=self.epsilon,
                    action_space=self.action_space,
                )
                action_pair = (actions_qa, self.action2str[action])
                (
                    observations,
                    reward,
                    done,
                    truncated,
                    info,
                ) = self.env.step(action_pair)
                done = done or truncated

                observations["room"] = self.manage_agent_and_map_memory(
                    observations["room"]
                )

                for obs in observations["room"]:
                    encode_observation(self.memory_systems, obs)
                    manage_memory(
                        self.memory_systems,
                        self.mm_policy,
                        self.mm_policy_model,
                        split_possessive=False,
                    )

                next_state = self.memory_systems.return_as_a_dict_list()
                transition = [state, action, reward, next_state, done]
                self.replay_buffer.store(*transition)

                if done or len(self.replay_buffer) >= self.warm_start:
                    break

    def train(self) -> None:
        """Train the explore agent."""
        self.fill_replay_buffer()  # fill up the buffer till warm start size
        super().train()
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
                self.init_memory_systems()
                observations, info = self.env.reset()

                observations["room"] = self.manage_agent_and_map_memory(
                    observations["room"]
                )

                for obs in observations["room"]:
                    encode_observation(self.memory_systems, obs)
                    manage_memory(
                        self.memory_systems,
                        self.mm_policy,
                        self.mm_policy_model,
                        split_possessive=False,
                    )

            actions_qa = [
                answer_question(
                    self.memory_systems,
                    self.qa_policy,
                    question,
                    split_possessive=False,
                )
                for question in observations["questions"]
            ]

            state = self.memory_systems.return_as_a_dict_list()
            action, q_values_ = select_action(
                state=state,
                greedy=False,
                dqn=self.dqn,
                epsilon=self.epsilon,
                action_space=self.action_space,
            )
            self.q_values["train"].append(q_values_)

            action_pair = (actions_qa, self.action2str[action])
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
                observations["room"] = self.manage_agent_and_map_memory(
                    observations["room"]
                )

                for obs in observations["room"]:
                    encode_observation(self.memory_systems, obs)
                    manage_memory(
                        self.memory_systems,
                        self.mm_policy,
                        self.mm_policy_model,
                        split_possessive=False,
                    )
                next_state = self.memory_systems.return_as_a_dict_list()
                transition = [state, action, reward, next_state, done]
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
        """A function shared by explore validation and test in the middle.

        Args:
            val_or_test: "val" or "test"

        Returns:
            scores_temp = a list of total episde rewards
            states = memory states
            q_values = q values
            actions = greey actions taken

        """
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

            self.init_memory_systems()
            observations, info = self.env.reset()

            observations["room"] = self.manage_agent_and_map_memory(
                observations["room"]
            )

            for obs in observations["room"]:
                encode_observation(self.memory_systems, obs)
                manage_memory(
                    self.memory_systems,
                    self.mm_policy,
                    self.mm_policy_model,
                    split_possessive=False,
                )

            while True:
                actions_qa = [
                    answer_question(
                        self.memory_systems,
                        self.qa_policy,
                        question,
                        split_possessive=False,
                    )
                    for question in observations["questions"]
                ]
                state = self.memory_systems.return_as_a_dict_list()
                if save_results:
                    states.append(deepcopy(state))

                action, q_values_ = select_action(
                    state=state,
                    greedy=True,
                    dqn=self.dqn,
                    epsilon=self.epsilon,
                    action_space=self.action_space,
                )
                if save_results:
                    q_values.append(q_values_)
                    actions.append(action)
                    self.q_values[val_or_test].append(q_values_)

                action_pair = (actions_qa, self.action2str[action])
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

                observations["room"] = self.manage_agent_and_map_memory(
                    observations["room"]
                )

                for obs in observations["room"]:
                    encode_observation(self.memory_systems, obs)
                    manage_memory(
                        self.memory_systems,
                        self.mm_policy,
                        self.mm_policy_model,
                        split_possessive=False,
                    )
            scores_temp.append(score)

        return scores_temp, states, q_values, actions
