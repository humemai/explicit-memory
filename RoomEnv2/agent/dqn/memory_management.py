"""Memory management Agent for the RoomEnv2 environment."""
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
    SemanticMemory,
    ShortMemory,
    MemorySystems,
)
from explicit_memory.nn import LSTM
from explicit_memory.policy import (
    answer_question,
    encode_observation,
    manage_memory,
    explore,
)
from explicit_memory.utils import ReplayBuffer, is_running_notebook, write_yaml
from .dqn import DQNAgent


class DQNMemoryManagementAgent(DQNAgent):
    """DQN Agent interacting with environment.

    Based on https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/01.dqn.ipynb
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
            "semantic": 16,
            "short": 1,
        },
        pretrain_semantic: bool = False,
        nn_params: dict = {
            "hidden_size": 64,
            "num_layers": 2,
            "n_actions": 5,
            "embedding_dim": 32,
            "v1_params": None,
            "v2_params": {},
        },
        run_validation: bool = True,
        run_test: bool = True,
        num_samples_for_results: int = 10,
        plotting_interval: int = 10,
        train_seed: int = 5,
        test_seed: int = 0,
        device: str = "cpu",
        memory_management_policy: str = "rl",
        qa_policy: str = "episodic_semantic",
        explore_policy: str = "avoid_walls",
        room_size: str = "dev",
        question_prob: float = 1.0,
        terminates_at: int = 99,
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
        memory_management_policy: Memory management policy. Choose one of "generalize",
            "random", "rl", or "neural"
        qa_policy: question answering policy Choose one of "episodic_semantic",
            "random", "rl", or "neural"
        explore_policy: The room exploration policy. Choose one of "random",
            "avoid_walls", "rl", or "neural"
        room_size: The room configuration to use. Choose one of "dev", "xxs", "xs",
            "s", "m", or "l".
        question_prob: The probability of a question being asked at every observation.
        terminates_at: The maximum number of steps to take in an episode.

        """
        self.all_params = deepcopy(locals())
        del self.all_params["self"]
        del self.all_params["__class__"]
        super().__init__(**self.all_params)
        assert self.memory_management_policy == " rl"
        assert self.qa_policy != "rl"
        assert self.explore_policy != "rl"

        self.action_space = gym.spaces.Discrete(3)  # episodic, semantic, or forget

    def step(
        self, action_manage_memory: int, question: List[str]
    ) -> Tuple[int, bool, List[str]]:
        """Take an action and return the response.

        Args
        ----
        action_manage_memory: This is the action that the agent takes, i.e.,
            manage memory systems.

        Returns
        -------
        reward: The reward for the action.
        done: Whether or not the episode ends.
        next_question: next question

        """
        assert not self.memory_systems.short.is_empty
        if action_manage_memory == 0:
            manage_memory(self.memory_systems, "episodic")
        elif action_manage_memory == 1:
            manage_memory(self.memory_systems, "semantic")
        elif action_manage_memory == 2:
            manage_memory(self.memory_systems, "forget")
        else:
            raise ValueError

        action_qa = self._answer_question(question)
        action_explore = self._explore()
        (
            (observations, next_question),
            reward,
            done,
            truncated,
            info,
        ) = self.env.step((action_qa, self.action_explore2str[action_explore]))

        for obs in observations:
            self._encode_observation(obs)
            self._manage_memory()
        done = done or truncated

        return reward, done, next_question

    def fill_replay_buffer(self) -> None:
        """Fill up the replay buffer in the beginning with the uniformly-sampled
        actions. The filling continues until it reaches the warm start size.

        Make sure that self.epsilon is 1.0 before calling this function.

        """
        print("filling up the replay buffer with warm start ...")

        self.dqn.eval()

        while len(self.replay_buffer) < self.warm_start:
            self.init_memory_systems()
            (observations, question), info = self.env.reset()
            for obs in observations:
                self._encode_observation(obs)

                done = False
                while not done and len(self.replay_buffer) < self.warm_start:
                    state = self.memory_systems.return_as_a_dict_list()
                    action_explore = self._explore()
                    reward, done, next_question = self.step(action_explore, question)
                    question = next_question
                    next_state = self.memory_systems.return_as_a_dict_list()
                    transition = [state, action_explore, reward, next_state, done]
                    self.replay_buffer.store(*transition)

        self.dqn.train()

        print("filling up the replay buffer done!")

    def train(self):
        """Train the agent."""
        self.fill_replay_buffer()  # fill up the buffer till warm start size
        self.num_validation = 0

        self.epsilons = []
        self.training_loss = []
        self.scores = {"train": [], "validation": [], "test": None}

        self.init_memory_systems()
        (observations, question), info = self.env.reset()
        for obs in observations:
            self._encode_observation(obs)
            self._manage_memory()

        score = 0
        bar = trange(1, self.num_iterations + 1)
        for self.iteration_idx in bar:
            state = self.memory_systems.return_as_a_dict_list()
            action_explore = self.select_action(state, greedy=False)
            reward, done, next_question = self.step(action_explore, question)
            question = next_question
            next_state = self.memory_systems.return_as_a_dict_list()
            score += reward
            transition = [state, action_explore, reward, next_state, done]
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
                for obs in observations:
                    self._encode_observation(obs)
                    self._manage_memory()

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
                self._plot()

        with torch.no_grad():
            self.test()

        self.env.close()

    def validate(self) -> None:
        """Validate the agent."""
        self.dqn.eval()

        scores = []
        for _ in range(self.num_samples_for_results):
            self.init_memory_systems()
            (observations, question), info = self.env.reset()
            for obs in observations:
                self._encode_observation(obs)
                self._manage_memory()

            done = False
            score = 0
            while not done:
                state = self.memory_systems.return_as_a_dict_list()
                action_explore = self.select_action(state, greedy=True)
                reward, done, next_question = self.step(action_explore, question)
                question = next_question
                score += reward
            scores.append(score)

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
        """Test the agent.

        Args
        ----
        checkpoint: The checkpoint to load the model from. If None, the model from the
            best validation is used.

        """
        env_config_test = self.env_config.copy()
        env_config_test["seed"] = self.test_seed
        self.env = gym.make(self.env_str, **env_config_test)
        self.dqn.eval()

        if self.run_validation:
            assert len(self.val_filenames) == 1
            self.dqn.load_state_dict(torch.load(self.val_filenames[0]))
            if checkpoint is not None:
                self.dqn.load_state_dict(torch.load(checkpoint))

        scores = []
        for _ in range(self.num_samples_for_results):
            self.init_memory_systems()
            (observations, question), info = self.env.reset()
            for obs in observations:
                self._encode_observation(obs)
                self._manage_memory()

            done = False
            score = 0
            while not done:
                state = self.memory_systems.return_as_a_dict_list()
                action_explore = self.select_action(state, greedy=True)
                reward, done, next_question = self.step(action_explore, question)
                question = next_question
                score += reward
            scores.append(score)

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
        write_yaml(self.all_params, os.path.join(self.default_root_dir, "train.yaml"))
        write_yaml(
            self.memory_systems.return_as_a_dict_list(),
            os.path.join(self.default_root_dir, "last_memory_state.yaml"),
        )

        self._plot()
        self.env.close()
        self.dqn.train()
