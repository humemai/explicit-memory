import os
from typing import Dict, List, Tuple
import random
from copy import deepcopy
import datetime

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from IPython.display import clear_output
from tqdm.auto import tqdm, trange

from explicit_memory.memory import EpisodicMemory, SemanticMemory, ShortMemory
from explicit_memory.policy import (
    answer_question,
    encode_observation,
    manage_memory,
)


from explicit_memory.utils import (
    ReplayBufferNStep,
    write_yaml,
    is_running_notebook,
    PrioritizedReplayBuffer,
)
from nn import LSTM


class HandcraftedAgent:

    """Handcrafted agent interacting with environment. This agent is not trained.
    Only one of the three agents, i.e., random, episodic_only, and semantic_only are
    suported
    """

    def __init__(
        self,
        env_str: str = "room_env:RoomEnv1-v1",
        policy: str = "random",
        num_samples_for_results: int = 10,
        test_seed: int = 42,
        capacity: dict = {
            "episodic": 16,
            "semantic": 16,
            "short": 1,
        },
    ):
        """Initialization.

        Args
        ----
        env_str: This has to be "room_env:RoomEnv1-v1"
        policy: The memory management policy. Choose one of "random", "episodic_only",
                or "semantic_only".
        num_samples_for_results: The number of samples to validate / test the agent.
        test_seed: The random seed for test.
        capacity: The capacity of each human-like memory systems.

        """
        self.all_params = deepcopy(locals())
        del self.all_params["self"]
        self.env_str = env_str
        self.policy = policy
        assert self.policy in ["random", "episodic_only", "semantic_only"]
        self.num_samples_for_results = num_samples_for_results
        self.test_seed = test_seed
        self.capacity = capacity

        self.env = gym.make(self.env_str, seed=self.test_seed)

        self.default_root_dir = f"./training_results/{str(datetime.datetime.now())}"
        os.makedirs(self.default_root_dir, exist_ok=True)

    def init_memory_systems(self, num_actions: int = 3) -> None:
        """Initialize the agent's memory systems. This has nothing to do with the
        replay buffer."""
        self.action_space = gym.spaces.Discrete(num_actions)
        self.memory_systems = {
            "episodic": EpisodicMemory(capacity=self.capacity["episodic"]),
            "semantic": SemanticMemory(capacity=self.capacity["semantic"]),
            "short": ShortMemory(capacity=self.capacity["short"]),
        }

    def find_answer(self) -> str:
        """Find an answer to the question, by looking up the memory systems."""
        if self.policy.lower() == "random":
            qa_policy = "episodic_semantic"
        elif self.policy.lower() == "episodic_only":
            qa_policy = "episodic"
        elif self.policy.lower() == "semantic_only":
            qa_policy = "semantic"
        else:
            raise ValueError("Unknown policy.")

        answer = answer_question(self.memory_systems, qa_policy, self.question)

        return str(answer).lower()

    def test(self):
        """Test the agent. There is no training for this agent, since it is
        handcrafted."""
        self.scores = []
        for _ in range(self.num_samples_for_results):
            self.init_memory_systems()
            (observation, self.question), info = self.env.reset()
            encode_observation(self.memory_systems, observation)

            done = False
            score = 0
            while not done:
                if self.policy.lower() == "random":
                    selected_action = random.choice(["episodic", "semantic", "forget"])
                    manage_memory(self.memory_systems, selected_action)
                elif self.policy.lower() == "episodic_only":
                    manage_memory(self.memory_systems, "episodic")
                elif self.policy.lower() == "semantic_only":
                    manage_memory(self.memory_systems, "semantic")
                else:
                    raise ValueError("Unknown policy.")

                answer = self.find_answer()
                (
                    (observation, self.question),
                    reward,
                    done,
                    truncated,
                    info,
                ) = self.env.step(answer)

                encode_observation(self.memory_systems, observation)
                score += reward
            self.scores.append(score)

        results = {
            "test_score": {
                "mean": round(np.mean(self.scores).item(), 2),
                "std": round(np.std(self.scores).item(), 2),
            }
        }
        write_yaml(results, os.path.join(self.default_root_dir, "results.yaml"))
        write_yaml(self.all_params, os.path.join(self.default_root_dir, "train.yaml"))


class DQNAgent:
    """DQN Agent interacting with environment.

    Based on https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/08.rainbow.ipynb
    """

    def __init__(
        self,
        env_str: str = "room_env:RoomEnv1-v1",
        num_iterations: int = 1280,
        replay_buffer_size: int = 1024,
        batch_size: int = 1024,
        target_update_rate: int = 10,
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
            "n_actions": 3,
            "embedding_dim": 32,
            "include_human": "sum",
        },
        run_validation: bool = True,
        run_test: bool = True,
        num_samples_for_results: int = 10,
        plotting_interval: int = 10,
        train_seed: int = 42,
        test_seed: int = 42,
        device: str = "cpu",
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        # Categorical DQN parameters
        v_min: float = 0.0,
        v_max: float = 200.0,
        atom_size: int = 51,
        # N-step Learning
        n_step: int = 3,
    ):
        """Initialization.

        Args
        ----
        env_str: This has to be "room_env:RoomEnv1-v1"
        num_iterations: The number of iterations to train the agent.
        replay_buffer_size: The size of the replay buffer.
        batch_size: The batch size for training This is the amount of samples sampled
            from the replay buffer.
        target_update_rate: The rate to update the target network.
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
        alpha: The alpha parameter for PER.
        beta: The beta parameter for PER.
        prior_eps: The epsilon parameter for PER.
        v_min: The minimum value for the categorical DQN.
        v_max: The maximum value for the categorical DQN.
        atom_size: The number of atoms for the categorical DQN.
        n_step: The number of steps for N-step Learning.

        """
        self.all_params = deepcopy(locals())
        del self.all_params["self"]

        self.is_notebook = is_running_notebook()
        self.env_str = env_str
        self.num_iterations = num_iterations
        self.plotting_interval = plotting_interval
        self.train_seed = train_seed
        self.test_seed = test_seed
        self.run_validation = run_validation
        if self.run_validation:
            self.default_root_dir = f"./training_results/{str(datetime.datetime.now())}"
            os.makedirs(self.default_root_dir, exist_ok=True)
            self.val_filenames = []
        self.run_test = run_test
        self.num_samples_for_results = num_samples_for_results
        self.device = torch.device(device)
        print(f"Running on {self.device}")

        self.env = gym.make(self.env_str, seed=self.train_seed)
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.target_update_rate = target_update_rate
        self.gamma = gamma

        # NoisyNet: All attributes related to epsilon are removed

        # PER
        # memory for 1-step Learning
        self.alpha = alpha
        self.beta = beta
        self.prior_eps = prior_eps

        # N-step Learning
        self.n_step = n_step

        self.replay_buffer = PrioritizedReplayBuffer(
            observation_type="dict",
            size=self.replay_buffer_size,
            batch_size=self.batch_size,
            alpha=self.alpha,
            n_step=self.n_step,
            gamma=self.gamma,
        )

        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.replay_buffer_n = ReplayBufferNStep(
                observation_type="dict",
                size=self.replay_buffer_size,
                batch_size=self.batch_size,
                n_step=self.n_step,
                gamma=self.gamma,
            )

        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(
            self.device
        )

        self.capacity = capacity
        self.nn_params = nn_params
        self.nn_params["capacity"] = self.capacity
        self.nn_params["device"] = self.device
        self.nn_params["entities"] = {
            "humans": self.env.des.humans,
            "objects": self.env.des.objects,
            "object_locations": self.env.des.object_locations,
        }
        self.nn_params["atom_size"] = self.atom_size
        self.nn_params["support"] = self.support

        # networks: dqn, dqn_target
        self.dqn = LSTM(**self.nn_params)
        self.dqn_target = LSTM(**self.nn_params)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in replay buffer
        self.transition = list()

        # mode: train / test
        self.is_test = False

        self.pretrain_semantic = pretrain_semantic

    def find_answer(self) -> str:
        """Find an answer to the question.

        This is hard coded. It's a "symbolic" reasoner. It first looks up the episodic
        memory systems, and if it cannot find the answer, it looks up the semantic
        memory systems.
        """
        answer = answer_question(
            self.memory_systems, "episodic_semantic", self.question
        )

        return str(answer).lower()

    def get_memory_state(self) -> dict:
        """Return the current state of the memory systems. This is NOT what the gym env
        gives you. This is made by the agent.
        """

        return {
            "episodic": self.memory_systems["episodic"].return_as_dicts(),
            "semantic": self.memory_systems["semantic"].return_as_dicts(),
            "short": self.memory_systems["short"].return_as_dicts(),
        }

    def select_action(self, state: dict) -> np.ndarray:
        """Select an action from the input state.

        NoisyNet: no epsilon greedy action selection

        Args
        ----
        state: The current state of the memory systems. This is NOT what the gym env
        gives you. This is made by our agent.

        """
        selected_action = self.dqn(state).argmax()
        selected_action = selected_action.detach().cpu().numpy()

        if not self.is_test:
            self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response.

        Args
        ----
        action: This action has nothing to do with the action that you give to the
        environment. This is the action that the agent takes, i.e., manage memory.

        Returns
        -------
        reward: The reward for the action.
        done: Whether or not the episode ends.

        """
        if action == 0:
            manage_memory(self.memory_systems, "episodic")
        elif action == 1:
            manage_memory(self.memory_systems, "semantic")
        elif action == 2:
            manage_memory(self.memory_systems, "forget")
        else:
            raise ValueError

        answer = self.find_answer()

        (observation, self.question), reward, done, truncated, info = self.env.step(
            answer
        )
        encode_observation(self.memory_systems, observation)
        done = done or truncated
        next_state = self.get_memory_state()

        if not self.is_test:
            self.transition += [reward, next_state, done]

            # N-step transition
            if self.use_n_step:
                one_step_transition = self.replay_buffer_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            if one_step_transition:
                self.replay_buffer.store(*one_step_transition)

        return reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.replay_buffer.sample_batch(self.beta)
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
        indices = samples["indices"]

        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)

        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)

        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma_ = self.gamma**self.n_step
            samples = self.replay_buffer_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma_)
            elementwise_loss += elementwise_loss_n_loss

            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.replay_buffer.update_priorities(indices, new_priorities)

        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()

    def fill_replay_buffer(self) -> None:
        """Make the replay buffer full in the beginning with the uniformly-sampled
        actions. The filling continues until it reaches the batch size."""

        self.is_test = False
        self.dqn.eval()
        while len(self.replay_buffer) < self.batch_size:
            self.init_memory_systems()
            (observation, self.question), info = self.env.reset()
            encode_observation(self.memory_systems, observation)

            done = False
            while not done and len(self.replay_buffer) < self.batch_size:
                state = self.get_memory_state()
                action = self.select_action(state)
                reward, done = self.step(action)

        self.dqn.train()

    def train(self):
        """Train the agent."""
        self.fill_replay_buffer()  # fill up the buffer till batch size
        self.is_test = False
        self.num_validation = 0

        self.init_memory_systems()
        (observation, self.question), info = self.env.reset()
        encode_observation(self.memory_systems, observation)

        self.training_loss = []
        self.scores = {"train": [], "validation": [], "test": None}

        score = 0
        bar = trange(1, self.num_iterations + 1)
        for self.iteration_idx in bar:
            state = self.get_memory_state()
            action = self.select_action(state)
            reward, done = self.step(action)

            score += reward

            # PER: increase beta
            fraction = min(self.iteration_idx / self.num_iterations, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # if episode ends
            if done:
                self.scores["train"].append(score)
                score = 0
                if self.run_validation:
                    with torch.no_grad():
                        self.validate()

                self.init_memory_systems()
                (observation, self.question), info = self.env.reset()
                encode_observation(self.memory_systems, observation)

            loss = self.update_model()
            self.training_loss.append(loss)

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

    def choose_best_val(self, filenames: list):
        scores = []
        for filename in filenames:
            scores.append(int(filename.split("val-score=")[-1].split(".pt")[0]))
        return filenames[scores.index(max(scores))]

    def validate(self) -> None:
        """Validate the agent."""
        self.is_test = True
        self.dqn.eval()

        scores = []
        for _ in range(self.num_samples_for_results):
            self.init_memory_systems()
            (observation, self.question), info = self.env.reset()
            encode_observation(self.memory_systems, observation)

            done = False
            score = 0
            while not done:
                state = self.get_memory_state()
                action = self.select_action(state)
                reward, done = self.step(action)

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
        self.is_test = False
        self.dqn.train()

    def test(self, checkpoint: str = None) -> None:
        """Test the agent.

        Args
        ----
        checkpoint: The checkpoint to load the model from. If None, the model from the
            best validation is used.

        """
        self.is_test = True
        self.env = gym.make(self.env_str, seed=self.test_seed)
        self.dqn.eval()

        if self.run_validation:
            assert len(self.val_filenames) == 1
            self.dqn.load_state_dict(torch.load(self.val_filenames[0]))
            if checkpoint is not None:
                self.dqn.load_state_dict(torch.load(checkpoint))

        scores = []
        for _ in range(self.num_samples_for_results):
            self.init_memory_systems()
            (observation, self.question), info = self.env.reset()
            encode_observation(self.memory_systems, observation)

            done = False
            score = 0
            while not done:
                state = self.get_memory_state()
                action = self.select_action(state)
                reward, done = self.step(action)

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

        self._plot()
        self.env.close()
        self.is_test = False
        self.dqn.train()

    def _compute_dqn_loss(
        self, samples: Dict[str, np.ndarray], gamma: float
    ) -> torch.Tensor:
        """Return dqn loss.

        Args
        ----
        samples: A dictionary of samples from the replay buffer.
            obs: np.ndarray,
            act: np.ndarray,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
        """
        state = samples["obs"]
        next_state = samples["next_obs"]
        action = torch.LongTensor(samples["acts"]).to(self.device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(self.device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(self.device)

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                )
                .long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _plot(self):
        """Plot the training progresses."""
        clear_output(True)
        plt.figure(figsize=(20, 8))

        if self.scores["train"]:
            plt.subplot(222)
            plt.title(
                f"iteration {self.iteration_idx} out of {self.num_iterations}. "
                f"training score: {self.scores['train'][-1]} out of 128"
            )
            plt.plot(self.scores["train"])
            plt.xlabel("episode")

        if self.scores["validation"]:
            plt.subplot(223)
            val_means = [
                round(np.mean(scores).item()) for scores in self.scores["validation"]
            ]
            plt.title(f"validation score: {val_means[-1]} out of 128")
            plt.plot(val_means)
            plt.xlabel("episode")

        if self.scores["test"]:
            plt.subplot(224)
            plt.title(f"test score: {np.mean(self.scores['test'])} out of 128")
            plt.plot(round(np.mean(self.scores["test"]).item(), 2))
            plt.xlabel("episode")

        plt.subplot(221)
        plt.title("training loss")
        plt.plot(self.training_loss)
        plt.xlabel("update counts")

        plt.subplots_adjust(hspace=0.5)
        plt.savefig(f"{self.default_root_dir}/plot.png")
        plt.show()

        if not self.is_notebook:
            self._console()

    def _console(self):
        """Print the training progresses to the console."""
        if self.scores["train"]:
            tqdm.write(
                f"iteration {self.iteration_idx} out of {self.num_iterations}.\n"
                f"episode {self.num_validation} training score: "
                f"{self.scores['train'][-1]} out of 128"
            )

        if self.scores["validation"]:
            val_means = [
                round(np.mean(scores).item()) for scores in self.scores["validation"]
            ]
            tqdm.write(
                f"episode {self.num_validation} validation score: {val_means[-1]} "
                "out of 128"
            )

        if self.scores["test"]:
            tqdm.write(f"test score: {np.mean(self.scores['test'])} out of 128")

        tqdm.write(f"training loss: {self.training_loss[-1]}\n")

    def init_memory_systems(self, num_actions: int = 3) -> None:
        """Initialize the agent's memory systems. This has nothing to do with the
        replay buffer."""
        self.action_space = gym.spaces.Discrete(num_actions)
        self.memory_systems = {
            "episodic": EpisodicMemory(capacity=self.capacity["episodic"]),
            "semantic": SemanticMemory(capacity=self.capacity["semantic"]),
            "short": ShortMemory(capacity=self.capacity["short"]),
        }

        if self.pretrain_semantic:
            assert self.capacity["semantic"] > 0
            _ = self.memory_systems["semantic"].pretrain_semantic(
                semantic_knowledge=self.env.des.semantic_knowledge,
                return_remaining_space=False,
                freeze=False,
            )
