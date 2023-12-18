"""DQN Agent for the RoomEnv2 environment."""
import os

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from room_env.envs.room2 import RoomEnv2

from explicit_memory.nn import LSTM
from explicit_memory.utils import (ReplayBuffer, is_running_notebook,
                                   plot_results, save_dqn_final_results,
                                   save_dqn_validation,
                                   save_states_q_values_actions, write_yaml)

from ..handcrafted import HandcraftedAgent


class DQNAgent(HandcraftedAgent):
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
            "semantic": 16,
            "short": 1,
        },
        pretrain_semantic: str | bool = False,
        nn_params: dict = {
            "hidden_size": 64,
            "num_layers": 2,
            "embedding_dim": 32,
            "v1_params": None,
            "v2_params": {},
        },
        run_test: bool = True,
        num_samples_for_results: int = 10,
        plotting_interval: int = 10,
        train_seed: int = 5,
        test_seed: int = 0,
        device: str = "cpu",
        mm_policy: str = "generalize",
        qa_policy: str = "episodic_semantic",
        explore_policy: str = "avoid_walls",
        env_config: dict = {
            "question_prob": 1.0,
            "terminates_at": 99,
            "room_size": "xxs",
            "randomize_observations": True,
        },
        ddqn: bool = False,
        dueling_dqn: bool = False,
        default_root_dir: str = "./training_results/",
        run_handcrafted_baselines: dict | None = None,
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
            default_root_dir: default root directory to store the results.
            run_handcrafted_baselines: Whether or not to run the handcrafted baselines.

        """
        self.train_seed = train_seed
        self.test_seed = test_seed
        env_config["seed"] = self.train_seed

        super().__init__(
            env_str=env_str,
            env_config=env_config,
            mm_policy=mm_policy,
            qa_policy=qa_policy,
            explore_policy=explore_policy,
            num_samples_for_results=num_samples_for_results,
            capacity=capacity,
            pretrain_semantic=pretrain_semantic,
            default_root_dir=default_root_dir,
        )

        self.device = torch.device(device)
        print(f"Running on {self.device}")

        self.ddqn = ddqn
        self.dueling_dqn = dueling_dqn

        self.nn_params = nn_params
        self.nn_params["capacity"] = self.capacity
        self.nn_params["device"] = self.device
        self.nn_params["entities"] = self.env.entities
        self.nn_params["relations"] = self.env.relations
        self.nn_params["dueling_dqn"] = self.dueling_dqn

        self.val_filenames = []
        self.is_notebook = is_running_notebook()
        self.num_iterations = num_iterations
        self.plotting_interval = plotting_interval
        self.run_test = run_test
        self.run_handcrafted_baselines = run_handcrafted_baselines

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
        self.dqn = LSTM(**self.nn_params)
        self.dqn_target = LSTM(**self.nn_params)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.replay_buffer = ReplayBuffer(
            observation_type="dict", size=replay_buffer_size, batch_size=batch_size
        )

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        self.q_values = {"train": [], "val": [], "test": []}

        if self.run_handcrafted_baselines is not None:
            self.run_and_save_handcrafted_baselines()

    def run_and_save_handcrafted_baselines(self) -> None:
        """Run and save the handcrafted baselines."""

        env = RoomEnv2(**self.env_config)
        observations, info = env.reset()
        env.render("image", save_fig_dir=self.default_root_dir)

        del env

        results = {}
        for policy in self.run_handcrafted_baselines:
            results[str(policy)] = []
            for test_seed in [0, 1, 2, 3, 4]:
                agent_handcrafted = HandcraftedAgent(
                    env_str="room_env:RoomEnv-v2",
                    env_config={**self.env_config, "seed": test_seed},
                    mm_policy=policy["mm"],
                    qa_policy=policy["qa"],
                    explore_policy=policy["explore"],
                    num_samples_for_results=10,
                    capacity=self.capacity,
                    pretrain_semantic=policy["pretrain_semantic"],
                    default_root_dir=self.default_root_dir,
                )
                agent_handcrafted.test()
                results[str(policy)].append(
                    agent_handcrafted.scores["test_score"]["mean"]
                )
                agent_handcrafted.remove_results_from_disk()
            results[str(policy)] = {
                "mean": np.mean(results[str(policy)]).item(),
                "std": np.std(results[str(policy)]).item(),
            }
        write_yaml(results, os.path.join(self.default_root_dir, "handcrafted.yaml"))

    def fill_replay_buffer(self) -> None:
        """Make the replay buffer full in the beginning with the uniformly-sampled
        actions. The filling continues until it reaches the warm start size.

        """
        pass

    def train(self) -> None:
        """Code for training"""

    def validate(self) -> None:
        self.dqn.eval()
        scores_temp, states, q_values, actions = self.validate_test_middle("val")

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

        save_dqn_final_results(
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
            self.env.total_maximum_episode_rewards,
            self.num_validation,
            self.num_samples_for_results,
            self.default_root_dir,
            to_plot,
            save_fig,
        )
