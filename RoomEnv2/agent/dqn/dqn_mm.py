"""DQN Agent for the RoomEnv2 environment."""
import datetime
import os
import random
import shutil
from copy import deepcopy
from typing import Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from tqdm.auto import tqdm, trange

from explicit_memory.memory import (
    EpisodicMemory,
    MemorySystems,
    SemanticMemory,
    ShortMemory,
)
from explicit_memory.nn import LSTM
from explicit_memory.policy import (
    answer_question,
    encode_observation,
    explore,
    manage_memory,
)
from explicit_memory.utils import ReplayBuffer, is_running_notebook, write_yaml

from .dqn import DQNAgent


class DQNMMAgent(DQNAgent):
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
        target_update_rate: int = 10,
        epsilon_decay_until: float = 2048,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.65,
        capacity: dict = {
            "episodic": 16,
            "episodic_agent": 16,
            "semantic": 16,
            "short": 1,
        },
        pretrain_semantic: bool = False,
        nn_params: dict = {
            "hidden_size": 64,
            "num_layers": 2,
            "embedding_dim": 32,
            "v1_params": None,
            "v2_params": {},
            "memory_of_interest": ["episodic", "episodic_agent", "semantic", "short"],
        },
        run_validation: bool = True,
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
            "room_size": "dev",
        },
        ddqn: bool = False,
        dueling_dqn: bool = False,
    ):
        """Initialization.

        Args
        ----
        env_str: This has to be "room_env:RoomEnv-v2"
        num_iterations: The number of iterations to train the agent.
        replay_buffer_size: The size of the replay buffer.
        warm_start: The number of samples to fill the replay buffer with, before
            starting
        batch_size: The batch size for training This is the amount of samples sampled
            from the replay buffer.
        target_update_rate: The rate to update the target network.
        epsilon_decay_until: The iteration index until which to decay epsilon.
        max_epsilon: The maximum epsilon.
        min_epsilon: The minimum epsilon.
        gamma: The discount factor.
        capacity: The capacity of each human-like memory systems.
        pretrain_semantic: Whether or not to pretrain the semantic memory system.
        nn_params: The parameters for the DQN (function approximator).
        run_validation: Whether or not to run validation.
        run_test: Whether or not to run test.
        num_samples_for_results: The number of samples to validate / test the agent.
        plotting_interval: The interval to plot the results.
        train_seed: The random seed for train.
        test_seed: The random seed for test.
        device: The device to run the agent on. This is either "cpu" or "cuda".
        qa_policy: question answering policy Choose one of "episodic_semantic",
            "random", or "neural". qa_policy shouldn't be trained with RL. There is no
            sequence of states / actions to learn from.
        explore_policy: The room exploration policy. Choose one of "random",
            "avoid_walls", "rl", or "neural"
        env_config: The configuration of the environment.
            question_prob: The probability of a question being asked at every
                observation.
            terminates_at: The maximum number of steps to take in an episode.
            seed: seed for env
            room_size: The room configuration to use. Choose one of "dev", "xxs", "xs",
                "s", "m", or "l".
        ddqn: wehther to use double dqn
        dueling_dqn: whether to use dueling dqn

        """
        all_params = deepcopy(locals())
        del all_params["self"]
        del all_params["__class__"]
        self.all_params = deepcopy(all_params)

        all_params["nn_params"]["n_actions"] = 3
        all_params["mm_policy"] = "rl"
        super().__init__(**all_params)

        write_yaml(self.all_params, os.path.join(self.default_root_dir, "train.yaml"))

        self.action2str = {0: "episodic", 1: "semantic", 2: "forget"}
        # action: 1. move to episodic, 2. move to semantic, 3. forget
        self.action_space = gym.spaces.Discrete(len(self.action2str))

    def fill_replay_buffer(self) -> None:
        """Make the replay buffer full in the beginning with the uniformly-sampled
        actions. The filling continues until it reaches the warm start size.

        """
        while len(self.replay_buffer) < self.warm_start:
            self.init_memory_systems()
            (observations, question), info = self.env.reset()
            obs = observations[0]
            encode_observation(self.memory_systems, obs)
            for obs in observations[1:]:
                if self.memory_systems.short.get_oldest_memory()[0] == "agent":
                    manage_memory(self.memory_systems, "agent")
                else:
                    state = self.memory_systems.return_as_a_dict_list()
                    action = self.select_action(state, greedy=False)
                    manage_memory(self.memory_systems, self.action2str[action])
                    encode_observation(self.memory_systems, obs)
                    next_state = self.memory_systems.return_as_a_dict_list()
                    transition = [state, action, 0, next_state, False]
                    self.replay_buffer.store(*transition)

            done = False
            while not done and len(self.replay_buffer) < self.warm_start:
                state = self.memory_systems.return_as_a_dict_list()
                action = self.select_action(state, greedy=False)
                manage_memory(self.memory_systems, self.action2str[action])
                action_qa = str(
                    answer_question(self.memory_systems, self.qa_policy, question)
                )
                action_explore = explore(self.memory_systems, self.explore_policy)
                action_pair = (action_qa, action_explore)
                (
                    (observations, question),
                    reward,
                    done,
                    truncated,
                    info,
                ) = self.env.step(action_pair)

                obs = observations[0]
                encode_observation(self.memory_systems, obs)
                done = done or truncated
                next_state = self.memory_systems.return_as_a_dict_list()
                transition = [state, action, reward, next_state, done]
                self.replay_buffer.store(*transition)
                for obs in observations[1:]:
                    state = self.memory_systems.return_as_a_dict_list()
                    action = self.select_action(state, greedy=False)
                    manage_memory(self.memory_systems, self.action2str[action])
                    encode_observation(self.memory_systems, obs)
                    next_state = self.memory_systems.return_as_a_dict_list()
                    transition = [state, action, 0, next_state, False]
                    self.replay_buffer.store(*transition)

    def train(self) -> None:
        """Train the memory management agent."""
        self.fill_replay_buffer()  # fill up the buffer till warm start size
        self.num_validation = 0

        self.epsilons = []
        self.training_loss = []
        self.scores = {"train": [], "validation": [], "test": None}

        self.dqn.train()
        self.init_memory_systems()
        (observations, question), info = self.env.reset()
        obs = observations[0]
        encode_observation(self.memory_systems, obs)
        for obs in observations[1:]:
            state = self.memory_systems.return_as_a_dict_list()
            action = self.select_action(state, greedy=False)
            manage_memory(self.memory_systems, self.action2str[action])
            encode_observation(self.memory_systems, obs)
            next_state = self.memory_systems.return_as_a_dict_list()
            transition = [state, action, 0, next_state, False]
            self.replay_buffer.store(*transition)

        score = 0
        bar = trange(1, self.num_iterations + 1)
        for self.iteration_idx in bar:
            state = self.memory_systems.return_as_a_dict_list()
            action = self.select_action(state, greedy=False)
            manage_memory(self.memory_systems, self.action2str[action])
            action_qa = str(
                answer_question(self.memory_systems, self.qa_policy, question)
            )
            action_explore = explore(self.memory_systems, self.explore_policy)
            action_pair = (action_qa, action_explore)
            (
                (observations, question),
                reward,
                done,
                truncated,
                info,
            ) = self.env.step(action_pair)
            score += reward

            obs = observations[0]
            encode_observation(self.memory_systems, obs)
            done = done or truncated
            next_state = self.memory_systems.return_as_a_dict_list()
            transition = [state, action, reward, next_state, done]
            self.replay_buffer.store(*transition)
            for obs in observations[1:]:
                state = self.memory_systems.return_as_a_dict_list()
                action = self.select_action(state, greedy=False)
                manage_memory(self.memory_systems, self.action2str[action])
                encode_observation(self.memory_systems, obs)
                next_state = self.memory_systems.return_as_a_dict_list()
                transition = [state, action, 0, next_state, False]
                self.replay_buffer.store(*transition)

            # if episode ends
            if done:
                self.scores["train"].append(score)
                score = 0
                if self.run_validation:
                    with torch.no_grad():
                        self.validate()

                self.init_memory_systems()
                (observations, question), info = self.env.reset()
                obs = observations[0]
                encode_observation(self.memory_systems, obs)
                for obs in observations[1:]:
                    state = self.memory_systems.return_as_a_dict_list()
                    action = self.select_action(state, greedy=False)
                    manage_memory(self.memory_systems, self.action2str[action])
                    encode_observation(self.memory_systems, obs)
                    next_state = self.memory_systems.return_as_a_dict_list()
                    transition = [state, action, 0, next_state, False]
                    self.replay_buffer.store(*transition)

            loss = self.update_model()
            self.training_loss.append(loss)

            # linearly decrease epsilon
            self.epsilon = max(
                self.min_epsilon,
                self.epsilon
                - (self.max_epsilon - self.min_epsilon) / self.epsilon_decay_until,
            )
            self.epsilons.append(self.epsilon)

            # if hard update is needed
            if self.iteration_idx % self.target_update_rate == 0:
                self._target_hard_update()

            # plotting & show training results
            if (
                self.iteration_idx == self.num_iterations
                or self.iteration_idx % self.plotting_interval == 0
            ):
                if self.is_notebook:
                    self._plot()
                else:
                    self._console()

        with torch.no_grad():
            self.test()

        self.env.close()

    def validate_test_middle(self) -> List[float]:
        """A function shared by validation and test in the middle.


        Returns
        -------
        scores: a list of scores
        """
        self.dqn.eval()
        scores = []
        for _ in range(self.num_samples_for_results):
            self.init_memory_systems()
            (observations, question), info = self.env.reset()
            obs = observations[0]
            encode_observation(self.memory_systems, obs)
            for obs in observations[1:]:
                state = self.memory_systems.return_as_a_dict_list()
                action = self.select_action(state, greedy=True)
                manage_memory(self.memory_systems, self.action2str[action])
                encode_observation(self.memory_systems, obs)

            done = False
            score = 0
            while not done:
                state = self.memory_systems.return_as_a_dict_list()
                action = self.select_action(state, greedy=False)
                manage_memory(self.memory_systems, self.action2str[action])
                action_qa = str(
                    answer_question(self.memory_systems, self.qa_policy, question)
                )
                action_explore = explore(self.memory_systems, self.explore_policy)
                action_pair = (action_qa, action_explore)
                (
                    (observations, question),
                    reward,
                    done,
                    truncated,
                    info,
                ) = self.env.step(action_pair)
                score += reward
                done = done or truncated

                obs = observations[0]
                encode_observation(self.memory_systems, obs)
                for obs in observations[1:]:
                    state = self.memory_systems.return_as_a_dict_list()
                    action = self.select_action(state, greedy=True)
                    manage_memory(self.memory_systems, self.action2str[action])
                    encode_observation(self.memory_systems, obs)

            scores.append(score)

        return scores

    def validate(self) -> None:
        """Validate the memory management agent."""

        scores = self.validate_test_middle()

        mean_score = round(np.mean(scores).item())
        filename = (
            f"{self.default_root_dir}/"
            f"episode={self.num_validation}_val-score={mean_score}.pt"
        )
        self.val_filenames.append(filename)
        torch.save(self.dqn.state_dict(), filename)
        self.scores["validation"].append(scores)

        file_to_keep = self.choose_best_val(self.val_filenames)

        for filename in self.val_filenames:
            if filename != file_to_keep:
                os.remove(filename)
                self.val_filenames.remove(filename)

        self.env.close()
        self.num_validation += 1
        self.dqn.train()

    def test(self, checkpoint: str = None) -> None:
        """Test the memory management agent.

        Args
        ----
        checkpoint: The checkpoint to load the model from. If None, the model from the
            best validation is used.

        """
        self.env = gym.make(self.env_str, seed=self.test_seed)

        if self.run_validation:
            assert len(self.val_filenames) == 1
            self.dqn.load_state_dict(torch.load(self.val_filenames[0]))
            if checkpoint is not None:
                self.dqn.load_state_dict(torch.load(checkpoint))

        scores = self.validate_test_middle()
        self.scores["test"] = scores

        results = {
            "train_score": self.scores["train"],
            "validation_score": [
                {
                    "mean": round(np.mean(scores).item(), 2),
                    "std": round(np.std(scores).item(), 2),
                }
                for scores in self.scores["validation"]
            ],
            "test_score": {
                "mean": round(np.mean(self.scores["test"]).item(), 2),
                "std": round(np.std(self.scores["test"]).item(), 2),
            },
            "training_loss": self.training_loss,
        }
        write_yaml(results, os.path.join(self.default_root_dir, "results.yaml"))
        write_yaml(
            self.memory_systems.return_as_a_dict_list(),
            os.path.join(self.default_root_dir, "last_memory_state.yaml"),
        )

        if self.is_notebook:
            self._plot()
        else:
            self._console()

        self.env.close()
        self.dqn.train()
