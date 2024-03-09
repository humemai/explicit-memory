"""PPO memory management agent for the RoomEnv2 environment."""

import os
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
from tqdm.auto import tqdm

from explicit_memory.utils import write_yaml
from explicit_memory.utils.ppo import (
    save_states_actions_probs_values,
    select_action,
    update_model,
)
from explicit_memory.policy import (
    answer_question,
    encode_observation,
    explore,
    manage_memory,
)

from .ppo import PPOAgent


class PPOMMAgent(PPOAgent):
    """PPO memory management Agent interacting with environment.

    Based on https://github.com/MrSyee/pg-is-all-you-need
    """

    def __init__(
        self,
        env_str: str = "room_env:RoomEnv-v2",
        num_episodes: int = 10,
        num_rollouts: int = 2,
        epoch_per_rollout: int = 64,
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
        default_root_dir: str = "./training_results/PPO/mm/LSTM/",
        run_handcrafted_baselines: bool = False,
    ) -> None:
        """Initialization.

        Args:
            env_str: environment string. This has to be "room_env:RoomEnv-v2"
            num_episodes: number of episodes
            num_rollouts: number of rollouts
            epoch_per_rollout: number of epochs per rollout
            batch_size: batch size
            gamma: discount factor
            tau: GAE parameter
            epsilon: PPO clip parameter
            entropy_weight: entropy weight
            capacity: The capacity of each human-like memory systems
            pretrain_semantic: whether to pretrain the semantic memory system.
            nn_params: neural network parameters
            run_test: whether to run test
            num_samples_for_results: The number of samples to validate / test the agent.
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

        Keep in mind that one step in the environment is equivalent to multiple steps
        in the memory management system. This is because the agent needs to take
        multiple steps in the memory management system to make one action in the
        environment.

        Args:
            observations: observations
            is_train_val_test: "train", "val", or "test"
            states_buffer: states buffer
            actions_buffer: actions buffer
            values_buffer: values buffer
            log_probs_buffer: log probs buffer
            append_states_actions_probs_values: whether to append states, actions,
                probs, and values to save the results.
            append_states: whether to append states to the buffer to save the results.

        Returns:
            reward: reward from the environment
            done: whether the episode is done from the environment
            observations: next observations from the environment

        """
        observations_ = self.manage_agent_and_map_memory(observations["room"])
        for obs in observations_:
            encode_observation(self.memory_systems, obs)

            state = self.memory_systems.return_as_a_dict_list()
            action, actor_probs, critic_value = select_action(
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

                self.actions_all[is_train_val_test].append(action)
                self.actor_probs_all[is_train_val_test].append(actor_probs)
                self.critic_values_all[is_train_val_test].append(critic_value)

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
                    self.init_memory_systems()
                    observations, info = self.env.reset()

                num_mm_actions = len(observations["room"])
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

                if self.split_reward_training:
                    reward = reward / num_mm_actions
                    rewards = [reward] * (num_mm_actions)
                else:
                    rewards = [0] * (num_mm_actions - 1) + [reward]

                for reward in rewards:
                    reward = np.reshape(reward, (1, -1)).astype(np.float64)
                    rewards_buffer.append(torch.FloatTensor(reward).to(self.device))

                dones = [False] * (num_mm_actions - 1) + [done]
                for done_ in dones:
                    done_ = np.reshape(done_, (1, -1))
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
            observations_ = self.manage_agent_and_map_memory(observations["room"])
            encode_observation(self.memory_systems, observations_[0])
            next_state = self.memory_systems.return_as_a_dict_list()

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
            self.init_memory_systems()
            observations, info = self.env.reset()

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
