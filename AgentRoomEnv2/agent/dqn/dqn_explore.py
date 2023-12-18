"""DQN Agent for the RoomEnv2 environment."""
import os
from copy import deepcopy

import gymnasium as gym
import torch
from tqdm.auto import trange

from explicit_memory.nn import LSTM
from explicit_memory.policy import (answer_question, encode_observation,
                                    manage_memory)
from explicit_memory.utils import (dqn_target_hard_update, select_dqn_action,
                                   update_dqn_model, write_yaml)

from .dqn import DQNAgent


class DQNExploreAgent(DQNAgent):
    """DQN Agent interacting with environment.

    Based on https://github.com/Curt-Park/rainbow-is-all-you-need/
    """

    def __init__(
        self,
        env_str: str = "room_env:RoomEnv-v2",
        num_iterations: int = 1000,
        replay_buffer_size: int = 102400,
        warm_start: int = 102400,
        batch_size: int = 1024,
        target_update_interval: int = 10,
        epsilon_decay_until: float = 2048,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.65,
        capacity: dict = {
            "episodic": 16,
            "episodic_agent": 0,
            "semantic": 16,
            "semantic_map": 0,
            "short": 1,
        },
        pretrain_semantic: bool = None,
        nn_params: dict = {
            "hidden_size": 64,
            "num_layers": 2,
            "embedding_dim": 32,
            "v1_params": None,
            "v2_params": {},
            "memory_of_interest": ["episodic", "episodic_agent", "semantic", "short"],
        },
        run_test: bool = True,
        num_samples_for_results: int = 10,
        plotting_interval: int = 10,
        train_seed: int = 5,
        test_seed: int = 0,
        device: str = "cpu",
        mm_policy: str = "generalize",
        qa_policy: str = "episodic_semantic",
        env_config: dict = {
            "question_prob": 1.0,
            "terminates_at": 99,
            "room_size": "xxs",
        },
        ddqn: bool = False,
        dueling_dqn: bool = False,
        default_root_dir: str = "./training_results/",
    ):
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

        """
        all_params = deepcopy(locals())
        del all_params["self"]
        del all_params["__class__"]
        self.all_params = deepcopy(all_params)

        all_params["nn_params"]["n_actions"] = 5
        all_params["explore_policy"] = "rl"
        super().__init__(**all_params)
        write_yaml(self.all_params, os.path.join(self.default_root_dir, "train.yaml"))

        self.action2str = {0: "north", 1: "east", 2: "south", 3: "west", 4: "stay"}
        self.action_space = gym.spaces.Discrete(len(self.action2str))

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
                    split_possessive=False,
                )

            while True:
                actions_qa = [
                    str(
                        answer_question(
                            self.memory_systems,
                            self.qa_policy,
                            question,
                            split_possessive=False,
                        )
                    )
                    for question in observations["questions"]
                ]
                state = self.memory_systems.return_as_a_dict_list()
                action, q_values_ = select_dqn_action(
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
        self.scores = {"train": [], "validation": [], "test": None}

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
                        split_possessive=False,
                    )
            actions_qa = [
                str(
                    answer_question(
                        self.memory_systems,
                        self.qa_policy,
                        question,
                        split_possessive=False,
                    )
                )
                for question in observations["questions"]
            ]

            state = self.memory_systems.return_as_a_dict_list()
            action, q_values_ = select_dqn_action(
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

            loss = update_dqn_model(
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
                dqn_target_hard_update(dqn=self.dqn, dqn_target=self.dqn_target)

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
                    split_possessive=False,
                )

            while True:
                actions_qa = [
                    str(
                        answer_question(
                            self.memory_systems,
                            self.qa_policy,
                            question,
                            split_possessive=False,
                        )
                    )
                    for question in observations["questions"]
                ]
                state = self.memory_systems.return_as_a_dict_list()
                if save_results:
                    states.append(deepcopy(state))

                action, q_values_ = select_dqn_action(
                    state=state,
                    greedy=True,
                    dqn=self.dqn,
                    epsilon=self.epsilon,
                    action_space=self.action_space,
                )
                if save_results:
                    q_values.append(deepcopy(q_values_))
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
                        split_possessive=False,
                    )
            scores_temp.append(score)

        return scores_temp, states, q_values, actions
