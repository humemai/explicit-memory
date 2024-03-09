"""DQN memory management Agent for the RoomEnv2 environment."""

import os
from copy import deepcopy

import gymnasium as gym
import torch
from tqdm.auto import trange

from explicit_memory.policy import (
    answer_question,
    encode_observation,
    explore,
    manage_memory,
)

from explicit_memory.utils import write_yaml

from explicit_memory.utils.dqn import target_hard_update, select_action, update_model

from .dqn import DQNAgent


class DQNMMAgent(DQNAgent):
    """DQN memory management Agent interacting with environment.

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
        pretrain_semantic: str | bool = False,
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
                "short",
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
        qa_policy: str = "episodic_semantic",
        explore_policy: str = "avoid_walls",
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
        split_reward_training: bool = False,
        default_root_dir: str = "./training_results/DQN/LSTM/mm",
        run_handcrafted_baselines: bool = False,
    ) -> None:
        """Initialization.

        Args:
            env_str: environment string. This has to be "room_env:RoomEnv-v2"
            num_iterations: number of iterations to train
            replay_buffer_size: size of replay buffer
            warm_start: number of steps to fill the replay buffer, before training
            batch_size: This is the amount of samples sampled from the replay buffer.
            target_update_interval: interval to update target network
            epsilon_decay_until: until which iteration to decay epsilon
            max_epsilon: maximum epsilon
            min_epsilon: minimum epsilon
            gamma: discount factor
            capacity: The capacity of each human-like memory systems
            pretrain_semantic: whether to pretrain the semantic memory system.
            nn_params: parameters for the neural network (DQN)
            run_test: whether to run test
            num_samples_for_results: The number of samples to validate / test the agent.
            plotting_interval: interval to plot results
            train_seed: seed for training
            test_seed: seed for testing
            device: This is either "cpu" or "cuda".
            mm_policy: memory management policy. Choose one of "generalize", "random",
                "rl", or "neural"
            qa_policy: question answering policy Choose one of "episodic_semantic",
                "random", or "neural". qa_policy shouldn't be trained with RL. There is
                no sequence of states / actions to learn from.
            explore_policy: The room exploration policy. Choose one of "random",
                "avoid_walls", "rl", or "neural"
            env_config: The configuration of the environment.
                question_prob: The probability of a question being asked at every
                    observation.
                terminates_at: The maximum number of steps to take in an episode.
                seed: seed for env
                room_size: The room configuration to use. Choose one of "dev", "xxs",
                    "xs", "s", "m", or "l".
            ddqn: whether to use double DQN
            dueling_dqn: whether to use dueling DQN
            split_reward_training: whether to split the rewards during training
            default_root_dir: default root directory to save results
            run_handcrafted_baselines: whether to run handcrafted baselines

        """
        all_params = deepcopy(locals())
        del all_params["self"]
        del all_params["__class__"]
        self.all_params = deepcopy(all_params)
        del all_params["split_reward_training"]
        self.split_reward_training = split_reward_training

        # action: 1. move to episodic, 2. move to semantic, 3. forget
        self.action2str = {0: "episodic", 1: "semantic", 2: "forget"}
        self.action_space = gym.spaces.Discrete(len(self.action2str))

        all_params["nn_params"]["n_actions"] = len(self.action2str)
        all_params["mm_policy"] = "rl"
        super().__init__(**all_params)
        write_yaml(self.all_params, os.path.join(self.default_root_dir, "train.yaml"))

    def fill_replay_buffer(self) -> None:
        """Make the replay buffer full in the beginning with the uniformly-sampled
        actions. The filling continues until it reaches the warm start size.

        """
        while len(self.replay_buffer) < self.warm_start:
            self.init_memory_systems()
            observations, info = self.env.reset()

            observations["room"] = self.manage_agent_and_map_memory(
                observations["room"]
            )

            obs = observations["room"][0]
            encode_observation(self.memory_systems, obs)
            transitions = []
            for obs in observations["room"][1:]:
                state = self.memory_systems.return_as_a_dict_list()
                action, q_values_ = select_action(
                    state=state,
                    greedy=False,
                    dqn=self.dqn,
                    epsilon=self.epsilon,
                    action_space=self.action_space,
                )
                manage_memory(
                    self.memory_systems, self.action2str[action], split_possessive=False
                )
                encode_observation(self.memory_systems, obs)
                next_state = self.memory_systems.return_as_a_dict_list()
                transitions.append([state, action, None, next_state, False])

            while True:
                state = self.memory_systems.return_as_a_dict_list()
                action, q_values_ = select_action(
                    state=state,
                    greedy=False,
                    dqn=self.dqn,
                    epsilon=self.epsilon,
                    action_space=self.action_space,
                )
                manage_memory(
                    self.memory_systems, self.action2str[action], split_possessive=False
                )
                actions_qa = [
                    answer_question(self.memory_systems, self.qa_policy, question)
                    for question in observations["questions"]
                ]
                action_explore = explore(self.memory_systems, self.explore_policy)
                action_pair = (actions_qa, action_explore)
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

                obs = observations["room"][0]
                encode_observation(self.memory_systems, obs)
                next_state = self.memory_systems.return_as_a_dict_list()
                transitions.append([state, action, None, next_state, done])

                for trans in transitions[:-1]:
                    if self.split_reward_training:
                        trans[2] = reward / len(transitions)
                    else:
                        trans[2] = 0
                    self.replay_buffer.store(*trans)

                trans = transitions[-1]
                if self.split_reward_training:
                    trans[2] = reward / len(transitions)
                else:
                    trans[2] = reward
                self.replay_buffer.store(*trans)

                if done or len(self.replay_buffer) >= self.warm_start:
                    break

                transitions = []
                for obs in observations["room"][1:]:
                    state = self.memory_systems.return_as_a_dict_list()
                    action, q_values_ = select_action(
                        state=state,
                        greedy=False,
                        dqn=self.dqn,
                        epsilon=self.epsilon,
                        action_space=self.action_space,
                    )
                    manage_memory(
                        self.memory_systems,
                        self.action2str[action],
                        split_possessive=False,
                    )
                    encode_observation(self.memory_systems, obs)
                    next_state = self.memory_systems.return_as_a_dict_list()
                    transitions.append([state, action, None, next_state, False])

    def train(self) -> None:
        """Train the memory management agent."""
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

                obs = observations["room"][0]
                encode_observation(self.memory_systems, obs)
                transitions = []
                for obs in observations["room"][1:]:
                    state = self.memory_systems.return_as_a_dict_list()
                    action, q_values_ = select_action(
                        state=state,
                        greedy=False,
                        dqn=self.dqn,
                        epsilon=self.epsilon,
                        action_space=self.action_space,
                    )
                    self.q_values["train"].append(q_values_)

                    manage_memory(
                        self.memory_systems,
                        self.action2str[action],
                        split_possessive=False,
                    )
                    encode_observation(self.memory_systems, obs)
                    next_state = self.memory_systems.return_as_a_dict_list()
                    transitions.append([state, action, None, next_state, False])

            state = self.memory_systems.return_as_a_dict_list()

            action, q_values_ = select_action(
                state=state,
                greedy=False,
                dqn=self.dqn,
                epsilon=self.epsilon,
                action_space=self.action_space,
            )
            self.q_values["train"].append(q_values_)

            manage_memory(
                self.memory_systems, self.action2str[action], split_possessive=False
            )

            actions_qa = [
                answer_question(self.memory_systems, self.qa_policy, question)
                for question in observations["questions"]
            ]

            action_explore = explore(self.memory_systems, self.explore_policy)
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

            obs = observations["room"][0]
            encode_observation(self.memory_systems, obs)
            next_state = self.memory_systems.return_as_a_dict_list()
            transitions.append([state, action, None, next_state, done])

            for trans in transitions[:-1]:
                if self.split_reward_training:
                    trans[2] = reward / len(transitions)
                else:
                    trans[2] = 0
                self.replay_buffer.store(*trans)

            trans = transitions[-1]
            if self.split_reward_training:
                trans[2] = reward / len(transitions)
            else:
                trans[2] = reward
            self.replay_buffer.store(*trans)

            if done:
                self.scores["train"].append(score)
                score = 0
                with torch.no_grad():
                    self.validate()

                training_episode_begins = True

            else:
                transitions = []
                for obs in observations["room"][1:]:
                    state = self.memory_systems.return_as_a_dict_list()
                    action, q_values_ = select_action(
                        state=state,
                        greedy=False,
                        dqn=self.dqn,
                        epsilon=self.epsilon,
                        action_space=self.action_space,
                    )
                    self.q_values["train"].append(q_values_)

                    manage_memory(
                        self.memory_systems,
                        self.action2str[action],
                        split_possessive=False,
                    )
                    encode_observation(self.memory_systems, obs)
                    next_state = self.memory_systems.return_as_a_dict_list()
                    transitions.append([state, action, None, next_state, False])

                training_episode_begins = False

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
        """A function shared by validation and test in the middle.

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

            obs = observations["room"][0]
            encode_observation(self.memory_systems, obs)
            for obs in observations["room"][1:]:
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

                manage_memory(
                    self.memory_systems, self.action2str[action], split_possessive=False
                )
                encode_observation(self.memory_systems, obs)

            while True:
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

                manage_memory(
                    self.memory_systems, self.action2str[action], split_possessive=False
                )
                actions_qa = [
                    answer_question(self.memory_systems, self.qa_policy, question)
                    for question in observations["questions"]
                ]
                action_explore = explore(self.memory_systems, self.explore_policy)
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

                obs = observations["room"][0]
                encode_observation(self.memory_systems, obs)

                if done:
                    break

                for obs in observations["room"][1:]:
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

                    manage_memory(
                        self.memory_systems,
                        self.action2str[action],
                        split_possessive=False,
                    )
                    encode_observation(self.memory_systems, obs)

            scores_temp.append(score)

        return scores_temp, states, q_values, actions
