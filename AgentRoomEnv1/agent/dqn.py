import datetime
import os
import random
import shutil
from copy import deepcopy

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
from explicit_memory.policy import answer_question, encode_observation, manage_memory
from explicit_memory.utils import (
    ReplayBuffer,
    dqn_target_hard_update,
    plot_results,
    save_dqn_final_results,
    save_dqn_validation,
    select_dqn_action,
    update_dqn_model,
    write_yaml,
    save_states_q_values_actions,
)

from .handcrafted import HandcraftedAgent


class DQNAgent(HandcraftedAgent):
    """DQN Agent interacting with environment.

    Based on https://github.com/Curt-Park/rainbow-is-all-you-need/
    """

    def __init__(
        self,
        env_str: str = "room_env:RoomEnv-v1",
        num_iterations: int = 128 * 16,
        replay_buffer_size: int = 1024,
        warm_start: int = 1024,
        batch_size: int = 1024,
        target_update_rate: int = 10,
        epsilon_decay_until: float = 128 * 16,
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
            "embedding_dim": 32,
            "v1_params": {
                "include_human": "sum",
                "human_embedding_on_object_location": False,
            },
            "v2_params": None,
        },
        run_test: bool = True,
        num_samples_for_results: int = 10,
        plotting_interval: int = 10,
        train_seed: int = 42,
        test_seed: int = 42,
        device: str = "cpu",
        ddqn: bool = False,
        dueling_dqn: bool = False,
        default_root_dir: str = "./training_results/",
        des_size: str = "l",
    ):
        """Initialization.

        Args:
            env_str: This has to be "room_env:RoomEnv-v1"
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
            run_test: Whether or not to run test.
            num_samples_for_results: The number of samples to validate / test the agent.
            plotting_interval: The interval to plot the results.
            train_seed: The random seed for train.
            test_seed: The random seed for test.
            device: The device to run the agent on. This is either "cpu" or "cuda".
            default_root_dir: default root directory to store the results.
            des_size: The size of the DES, e.g., xxs, xs, s, m, or, l

        """
        all_params = deepcopy(locals())
        del all_params["self"]
        del all_params["__class__"]
        self.all_params = deepcopy(all_params)
        super().__init__(
            env_str=env_str,
            policy="rl",
            num_samples_for_results=num_samples_for_results,
            seed=train_seed,
            capacity=capacity,
            pretrain_semantic=pretrain_semantic,
            default_root_dir=default_root_dir,
            des_size=des_size,
        )
        write_yaml(self.all_params, os.path.join(self.default_root_dir, "train.yaml"))

        self.train_seed = train_seed
        self.test_seed = test_seed

        self.val_filenames = []
        self.num_iterations = num_iterations
        self.plotting_interval = plotting_interval
        self.run_test = run_test
        self.device = torch.device(device)
        print(f"Running on {self.device}")

        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_until = epsilon_decay_until
        self.target_update_rate = target_update_rate
        self.gamma = gamma
        self.warm_start = warm_start
        assert self.batch_size <= self.warm_start <= self.replay_buffer_size

        self.action2str = {
            0: "episodic",
            1: "semantic",
            2: "forget",
        }
        self.action_space = gym.spaces.Discrete(len(self.action2str))

        self.ddqn = ddqn
        self.dueling_dqn = dueling_dqn

        self.nn_params = nn_params
        self.nn_params["capacity"] = self.capacity
        self.nn_params["device"] = self.device
        self.nn_params["entities"] = (
            self.env.des.humans + self.env.des.objects + self.env.des.object_locations
        )
        # there is only one relation in v1, so just ignore it.
        self.nn_params["relations"] = []

        self.nn_params["memory_of_interest"] = ["episodic", "semantic", "short"]
        self.nn_params["n_actions"] = len(self.action2str)
        self.nn_params["dueling_dqn"] = self.dueling_dqn

        # networks: dqn, dqn_target
        self.dqn = LSTM(**self.nn_params)
        self.dqn_target = LSTM(**self.nn_params)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.replay_buffer = ReplayBuffer(
            observation_type="dict", size=replay_buffer_size, batch_size=batch_size
        )

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        self.train_val_test = None
        self.q_values = {"train": [], "val": [], "test": []}

    def fill_replay_buffer(self) -> None:
        """Make the replay buffer full in the beginning with the uniformly-sampled
        actions. The filling continues until it reaches the warm start size."""
        self.dqn.eval()

        while len(self.replay_buffer) < self.warm_start:
            self.init_memory_systems()
            (observation, question), info = self.env.reset()
            encode_observation(self.memory_systems, observation)

            while True:
                state = self.memory_systems.return_as_a_dict_list()
                action, q_values_ = select_dqn_action(
                    state=state,
                    greedy=False,
                    dqn=self.dqn,
                    train_val_test=self.train_val_test,
                    epsilon=self.epsilon,
                    action_space=self.action_space,
                )
                manage_memory(
                    self.memory_systems, self.action2str[action], split_possessive=True
                )

                answer = str(
                    answer_question(self.memory_systems, "episodic_semantic", question)
                )

                (
                    (observation, question),
                    reward,
                    done,
                    truncated,
                    info,
                ) = self.env.step(answer)
                done = done or truncated

                encode_observation(self.memory_systems, observation)
                next_state = self.memory_systems.return_as_a_dict_list()

                transition = [state, action, reward, next_state, done]
                self.replay_buffer.store(*transition)

                if done or len(self.replay_buffer) >= self.warm_start:
                    break

        self.dqn.train()

    def train(self):
        """Train the agent."""
        self.fill_replay_buffer()  # fill up the buffer till warm start size
        self.train_val_test = "train"
        self.num_validation = 0

        self.epsilons = []
        self.training_loss = []
        self.scores = {"train": [], "validation": [], "test": None}

        self.init_memory_systems()
        (observation, question), info = self.env.reset()
        encode_observation(self.memory_systems, observation)

        score = 0
        bar = trange(1, self.num_iterations + 1)
        for self.iteration_idx in bar:
            state = self.memory_systems.return_as_a_dict_list()
            action, q_values_ = select_dqn_action(
                state=state,
                greedy=False,
                dqn=self.dqn,
                train_val_test=self.train_val_test,
                epsilon=self.epsilon,
                action_space=self.action_space,
            )
            self.q_values["train"].append(q_values_)

            manage_memory(
                self.memory_systems, self.action2str[action], split_possessive=True
            )

            answer = str(
                answer_question(self.memory_systems, "episodic_semantic", question)
            )

            (
                (observation, question),
                reward,
                done,
                truncated,
                info,
            ) = self.env.step(answer)
            score += reward

            encode_observation(self.memory_systems, observation)
            done = done or truncated
            next_state = self.memory_systems.return_as_a_dict_list()

            transition = [state, action, reward, next_state, done]
            self.replay_buffer.store(*transition)

            # if episode ends
            if done:
                self.scores["train"].append(score)
                score = 0
                with torch.no_grad():
                    self.validate()

                self.init_memory_systems()
                (observation, question), info = self.env.reset()
                encode_observation(self.memory_systems, observation)

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
            if self.iteration_idx % self.target_update_rate == 0:
                dqn_target_hard_update(dqn=self.dqn, dqn_target=self.dqn_target)

            # plotting & show training results
            if (
                self.iteration_idx == self.num_iterations
                or self.iteration_idx % self.plotting_interval == 0
            ):
                plot_results(
                    self.scores,
                    self.training_loss,
                    self.epsilons,
                    self.q_values,
                    self.iteration_idx,
                    self.action_space.n.item(),
                    self.num_iterations,
                    self.env.total_episode_rewards,
                    self.num_validation,
                    self.num_samples_for_results,
                    self.default_root_dir,
                    "all",
                    True,
                )
        with torch.no_grad():
            self.test()

        self.env.close()

    def validate(self) -> None:
        """Validate the agent."""
        self.train_val_test = "val"
        self.dqn.eval()

        scores_temp = []
        states = []
        q_values = []
        actions = []

        for idx in range(self.num_samples_for_results):
            self.init_memory_systems()
            (observation, question), info = self.env.reset()
            encode_observation(self.memory_systems, observation)

            done = False
            score = 0
            while not done:
                if idx == self.num_samples_for_results - 1:
                    save_results = True
                else:
                    save_results = False

                state = self.memory_systems.return_as_a_dict_list()
                if save_results:
                    states.append(deepcopy(state))

                action, q_values_ = select_dqn_action(
                    state=state,
                    greedy=True,
                    dqn=self.dqn,
                    train_val_test=self.train_val_test,
                    epsilon=self.epsilon,
                    action_space=self.action_space,
                )
                if save_results:
                    q_values.append(deepcopy(q_values_))
                    actions.append(action)
                    self.q_values["val"].append(q_values_)

                manage_memory(
                    self.memory_systems, self.action2str[action], split_possessive=True
                )

                answer = str(
                    answer_question(self.memory_systems, "episodic_semantic", question)
                )
                (
                    (observation, question),
                    reward,
                    done,
                    truncated,
                    info,
                ) = self.env.step(answer)
                score += reward

                encode_observation(self.memory_systems, observation)
                done = done or truncated

            scores_temp.append(score)

        save_dqn_validation(
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
        self.train_val_test = "train"

    def test(self, checkpoint: str = None) -> None:
        """Test the agent.

        Args:
            checkpoint: The checkpoint to load the model from. If None, the model from the
                best validation is used.

        """
        self.train_val_test = "test"
        self.env = gym.make(
            self.env_str, seed=self.test_seed, des_size=self.all_params["des_size"]
        )
        self.dqn.eval()

        states = []
        q_values = []
        actions = []

        assert len(self.val_filenames) == 1
        self.dqn.load_state_dict(torch.load(self.val_filenames[0]))
        if checkpoint is not None:
            self.dqn.load_state_dict(torch.load(checkpoint))

        scores = []
        for idx in range(self.num_samples_for_results):
            self.init_memory_systems()
            (observation, question), info = self.env.reset()
            encode_observation(self.memory_systems, observation)

            done = False
            score = 0
            while not done:
                if idx == self.num_samples_for_results - 1:
                    save_results = True
                else:
                    save_results = False

                state = self.memory_systems.return_as_a_dict_list()
                if save_results:
                    states.append(deepcopy(state))

                action, q_values_ = select_dqn_action(
                    state=state,
                    greedy=True,
                    dqn=self.dqn,
                    train_val_test=self.train_val_test,
                    epsilon=self.epsilon,
                    action_space=self.action_space,
                )
                if save_results:
                    q_values.append(deepcopy(q_values_))
                    actions.append(action)
                    self.q_values["test"].append(q_values_)

                manage_memory(
                    self.memory_systems, self.action2str[action], split_possessive=True
                )

                answer = str(
                    answer_question(self.memory_systems, "episodic_semantic", question)
                )
                (
                    (observation, question),
                    reward,
                    done,
                    truncated,
                    info,
                ) = self.env.step(answer)
                score += reward

                encode_observation(self.memory_systems, observation)
                done = done or truncated

            scores.append(score)

        self.scores["test"] = scores

        save_dqn_final_results(
            self.scores,
            self.training_loss,
            self.default_root_dir,
            self.q_values,
        )
        save_states_q_values_actions(
            states, q_values, actions, self.default_root_dir, "test"
        )

        plot_results(
            self.scores,
            self.training_loss,
            self.epsilons,
            self.q_values,
            self.iteration_idx,
            self.action_space.n.item(),
            self.num_iterations,
            self.env.total_episode_rewards,
            self.num_validation,
            self.num_samples_for_results,
            self.default_root_dir,
            "all",
            True,
        )
        self.env.close()
        self.dqn.train()
