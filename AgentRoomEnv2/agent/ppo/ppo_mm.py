"""PPO memory management agent for the RoomEnv2 environment."""

import os
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from room_env.envs.room2 import RoomEnv2
from tqdm import trange

from explicit_memory.nn import LSTM
from explicit_memory.utils.ppo import (
    save_states_actions_probs_values,
    select_action,
    update_model,
    save_validation,
    save_final_results,
    plot_results,
)
from explicit_memory.policy import (
    answer_question,
    encode_observation,
    explore,
    manage_memory,
)

from explicit_memory.utils import is_running_notebook, write_yaml

from .ppo import PPOAgent


class PPOMMAgent(PPOAgent):
    """PPO memory management Agent interacting with environment.

    Based on https://github.com/MrSyee/pg-is-all-you-need
    """

    def __init__(
        self,
        env_str: str = "room_env:RoomEnv-v2",
        num_episodes: int = 10,
        rollout_multiples: int = 2,
        epoch: int = 64,
        batch_size: int = 128,
        gamma: float = 0.9,
        tau: float = 0.8,
        epsilon: float = 0.2,
        entropy_weight: float = 0.005,
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
        split_reward_training: bool = False,
        default_root_dir: str = "./training_results/PPO",
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


        """
        all_params = deepcopy(locals())
        del all_params["self"]
        del all_params["__class__"]
        self.all_params = deepcopy(all_params)
        del all_params["split_reward_training"]
        self.split_reward_training = split_reward_training

        all_params["nn_params"]["n_actions"] = 3
        all_params["mm_policy"] = "rl"
        super().__init__(**all_params)
        write_yaml(self.all_params, os.path.join(self.default_root_dir, "train.yaml"))

        self.action2str = {0: "episodic", 1: "semantic", 2: "forget"}
        # action: 1. move to episodic, 2. move to semantic, 3. forget
        self.action_space = gym.spaces.Discrete(len(self.action2str))

    def train(self):
        """Train the agent."""

        self.num_validation = 0

        score = 0
        bar = trange(1, self.num_episodes + 1)
        for self.outer_loop_idx in bar:
            (
                states_buffer,
                actions_buffer,
                rewards_buffer,
                values_buffer,
                masks_buffer,
                log_probs_buffer,
            ) = self.create_empty_rollout_buffer()

            new_episode_starts = True
            episode_idx = 0
            num_mm_actions = 0
            for self.inner_loop_idx in range(
                1, (self.rollout_multiples * self.num_steps_in_episode) + 1
            ):

                if episode_idx == (self.rollout_multiples - 1):
                    is_last_episode = True
                else:
                    is_last_episode = False

                if new_episode_starts:
                    self.init_memory_systems()
                    observations, info = self.env.reset()

                    observations["room"] = self.manage_agent_and_map_memory(
                        observations["room"]
                    )
                    obs = observations["room"][0]
                    encode_observation(self.memory_systems, obs)

                    for obs in observations["room"][1:]:
                        state = self.memory_systems.return_as_a_dict_list()
                        action, actor_probs, critic_value = select_action(
                            actor=self.actor,
                            critic=self.critic,
                            state=state,
                            is_test=False,
                            states=states_buffer,
                            actions=actions_buffer,
                            values=values_buffer,
                            log_probs=log_probs_buffer,
                        )
                        num_mm_actions += 1

                        if is_last_episode:
                            # None is a placeholder.
                            self.states_all["train"].append(None)
                            self.actions_all["train"].append(action)
                            self.actor_probs_all["train"].append(actor_probs)
                            self.critic_values_all["train"].append(critic_value)

                        manage_memory(
                            self.memory_systems,
                            self.action2str[action],
                            split_possessive=False,
                        )
                        encode_observation(self.memory_systems, obs)

                state = self.memory_systems.return_as_a_dict_list()
                action, actor_probs, critic_value = select_action(
                    actor=self.actor,
                    critic=self.critic,
                    state=state,
                    is_test=False,
                    states=states_buffer,
                    actions=actions_buffer,
                    values=values_buffer,
                    log_probs=log_probs_buffer,
                )
                num_mm_actions += 1

                if is_last_episode:
                    self.states_all["train"].append(None)
                    self.actions_all["train"].append(action)
                    self.actor_probs_all["train"].append(actor_probs)
                    self.critic_values_all["train"].append(critic_value)

                manage_memory(
                    self.memory_systems,
                    self.action2str[action],
                    split_possessive=True,
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

                reward = np.reshape(reward, (1, -1)).astype(np.float64)
                done = np.reshape(done, (1, -1))

                if not self.split_reward_training:
                    for _ in range(num_mm_actions):
                        rewards_buffer.append(torch.FloatTensor(reward).to(self.device))
                        masks_buffer.append(torch.FloatTensor(1 - done).to(self.device))
                else:
                    raise NotImplementedError

                # if episode ends
                if done:
                    episode_idx += 1
                    if is_last_episode:
                        self.scores_all["train"].append(score)
                        with torch.no_grad():
                            self.validate()

                    score = 0
                    new_episode_starts = True
                else:

                    for obs in observations["room"][1:]:
                        state = self.memory_systems.return_as_a_dict_list()
                        action, actor_probs, critic_value = select_action(
                            actor=self.actor,
                            critic=self.critic,
                            state=state,
                            is_test=False,
                            states=states_buffer,
                            actions=actions_buffer,
                            values=values_buffer,
                            log_probs=log_probs_buffer,
                        )
                        if is_last_episode:
                            self.states_all["train"].append(None)
                            self.actions_all["train"].append(action)
                            self.actor_probs_all["train"].append(actor_probs)
                            self.critic_values_all["train"].append(critic_value)

                        manage_memory(
                            self.memory_systems,
                            self.action2str[action],
                            split_possessive=False,
                        )
                        encode_observation(self.memory_systems, obs)
                        next_state = self.memory_systems.return_as_a_dict_list()

                    new_episode_starts = False

            actor_loss, critic_loss = update_model(
                next_state,
                states_buffer,
                actions_buffer,
                rewards_buffer,
                values_buffer,
                masks_buffer,
                log_probs_buffer,
                self.gamma,
                self.tau,
                self.epoch,
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

            # plotting & show training results
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
            scores:

        """
        scores = []

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

                action, actor_probs, critic_value = select_action(
                    actor=self.actor,
                    critic=self.critic,
                    state=state,
                    is_test=True,
                )
                if save_results:
                    self.states_all["val"].append(deepcopy(state))
                    self.actions_all["val"].append(action)
                    self.actor_probs_all["val"].append(actor_probs)
                    self.critic_values_all["val"].append(critic_value)

                manage_memory(
                    self.memory_systems,
                    self.action2str[action],
                    split_possessive=True,
                )
                encode_observation(self.memory_systems, obs)

            while True:
                state = self.memory_systems.return_as_a_dict_list()

                action, actor_probs, critic_value = select_action(
                    actor=self.actor,
                    critic=self.critic,
                    state=state,
                    is_test=True,
                )
                if save_results:
                    self.states_all["val"].append(deepcopy(state))
                    self.actions_all["val"].append(action)
                    self.actor_probs_all["val"].append(actor_probs)
                    self.critic_values_all["val"].append(critic_value)

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

                    action, actor_probs, critic_value = select_action(
                        actor=self.actor,
                        critic=self.critic,
                        state=state,
                        is_test=True,
                    )
                    if save_results:
                        self.states_all["val"].append(deepcopy(state))
                        self.actions_all["val"].append(action)
                        self.actor_probs_all["val"].append(actor_probs)
                        self.critic_values_all["val"].append(critic_value)

                    manage_memory(
                        self.memory_systems,
                        self.action2str[action],
                        split_possessive=True,
                    )
                    encode_observation(self.memory_systems, obs)

            scores.append(score)

        return scores
