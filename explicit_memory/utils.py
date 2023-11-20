"""Utility functions."""
import csv
import json
import logging
import os
import pickle
import random
import shutil
from collections import deque
from copy import deepcopy
from glob import glob
from pprint import pformat
from typing import Deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from IPython.display import clear_output
from tqdm.auto import tqdm, trange

from .segment_tree import MinSegmentTree, SumSegmentTree

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def remove_timestamp(entry: list[str]) -> list:
    """Remove the timestamp from a given observation/episodic memory.

    Args:
        entry: An observation / episodic memory in a quadruple format
            (i.e., (head, relation, tail, timestamp))

    Returns:
        entry_without_timestamp: i.e., (head, relation, tail)

    """
    assert len(entry) == 4
    logging.debug(f"Removing timestamp from {entry} ...")
    entry_without_timestamp = entry[:-1]
    logging.info(f"Timestamp is removed from {entry}: {entry_without_timestamp}")

    return entry_without_timestamp


def split_by_possessive(name_entity: str) -> tuple[str, str]:
    """Separate name and entity from the given string.

    Args:
        name_entity: e.g., "tae's laptop"

    Returns:
        name: e.g., tae
        entity: e.g., laptop

    """
    logging.debug(f"spliting name and entity from {name_entity}")
    if "'s " in name_entity:
        name, entity = name_entity.split("'s ")
    else:
        name, entity = None, None

    return name, entity


def remove_posession(entity: str) -> str:
    """Remove name from the entity.

    Args:
        entity: e.g., bob's laptop

    Returns:
        e.g., laptop

    """
    return entity.split("'s ")[-1]


def seed_everything(seed: int) -> None:
    """Seed every randomness to seed"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def read_json(fname: str) -> dict:
    """Read json"""
    logging.debug(f"reading json {fname} ...")
    with open(fname, "r") as stream:
        return json.load(stream)


def write_json(content: dict, fname: str) -> None:
    """Write json"""
    logging.debug(f"writing json {fname} ...")
    with open(fname, "w") as stream:
        json.dump(content, stream, indent=4, sort_keys=False)


def read_yaml(fname: str) -> dict:
    """Read yaml."""
    logging.debug(f"reading yaml {fname} ...")
    with open(fname, "r") as stream:
        return yaml.safe_load(stream)


def write_yaml(content: dict, fname: str) -> None:
    """write yaml."""
    logging.debug(f"writing yaml {fname} ...")
    with open(fname, "w") as stream:
        yaml.dump(content, stream, indent=2, sort_keys=False)


def read_pickle(fname: str):
    """Read pickle"""
    logging.debug(f"writing pickle {fname} ...")
    with open(fname, "rb") as stream:
        foo = pickle.load(stream)
    return foo


def write_csv(content: list, fname: str) -> None:
    with open(fname, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerows(content)


def read_data(data_path: str) -> dict:
    """Read train, val, test spilts.

    Args:
        data_path: path to data.

    Returns:
        data: {'train': list of training obs,
            'val': list of val obs,
            'test': list of test obs}

    """
    logging.debug(f"reading data from {data_path} ...")
    data = read_json(data_path)
    logging.info(f"Succesfully read data {data_path}")

    return data


def load_questions(path: str) -> dict:
    """Load premade questions.

    Args:
        path: path to the question json file.

    """
    logging.debug(f"loading questions from {path}...")
    questions = read_json(path)
    logging.info(f"questions loaded from {path}!")

    return questions


def argmax(iterable):
    """argmax"""
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def get_duplicate_dicts(search: dict, target: list) -> list:
    """Find if there are duplicate dicts.

    Args:
        search: dict
        target: target list to look up.

    Returns:
        duplicates: a list of dicts or None

    """
    assert isinstance(search, dict)
    logging.debug("finding if duplicate dicts exist ...")
    duplicates = []

    for candidate in target:
        assert isinstance(candidate, dict)
        if set(search).issubset(set(candidate)):
            if all([val == candidate[key] for key, val in search.items()]):
                duplicates.append(candidate)

    logging.info(f"{len(duplicates)} duplicates were found!")

    return duplicates


def list_duplicates_of(seq, item) -> list:
    # https://stackoverflow.com/questions/5419204/index-of-duplicates-items-in-a-python-list
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item, start_at + 1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs


def rename_training_dirs(root_dir: str = "./training_results/"):
    old_dirs = []
    new_dirs = []
    for foo in glob(os.path.join(root_dir, "*")):
        bar = glob(os.path.join(foo, "*/*/*test*"))
        if len(bar) == 0:
            continue
        bar = bar[0]
        old_dir = "/".join(bar.split("/")[:-1])
        old_dirs.append(old_dir)
        hparams = read_yaml(os.path.join(old_dir, "hparams.yaml"))

        allow_random_human = hparams["allow_random_human"]
        allow_random_question = hparams["allow_random_question"]
        pretrain_semantic = hparams["pretrain_semantic"]
        varying_rewards = hparams["varying_rewards"]
        capacity = hparams["capacity"]["episodic"] + hparams["capacity"]["semantic"]
        question_prob = hparams["question_prob"]
        des_size = hparams["des_size"]
        seed = hparams["seed"]

        new_dir = (
            f"training_results/"
            f"allow_random_human={allow_random_human}_"
            f"allow_random_question={allow_random_question}_"
            f"pretrain_semantic={pretrain_semantic}_"
            f"varying_rewards={varying_rewards}_"
            f"des_size={des_size}_"
            f"capacity={capacity}_"
            f"question_prob={question_prob}_"
            f"seed={seed}"
        )
        new_dirs.append(new_dir)
        os.rename(old_dir, new_dir)

    for foo in glob(os.path.join(root_dir, "*/lightning_logs")):
        if len(os.listdir(foo)) == 0:
            dir_to_delete = os.path.dirname(foo)
            shutil.rmtree(dir_to_delete)


def is_running_notebook() -> bool:
    """See if the code is running in a notebook or not."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def plot_dqn(
    scores: dict,
    training_loss: list,
    epsilons: list,
    q_values: dict,
    iteration_idx: int,
    number_of_actions: int,
    num_iterations: int,
    total_episode_rewards: int,
    num_validation: int,
    num_samples_for_results: int,
    default_root_dir: str,
) -> None:
    """Plot things for DQN training."""
    all_params = deepcopy(locals())

    is_notebook = is_running_notebook()

    if is_notebook:
        clear_output(True)
    plt.figure(figsize=(20, 20))

    if scores["train"]:
        plt.subplot(334)
        plt.title(
            f"iteration {iteration_idx} out of {num_iterations}. "
            f"training score: {scores['train'][-1]} out of {total_episode_rewards}"
        )
        plt.plot(scores["train"])
        plt.xlabel("episode")

    if scores["validation"]:
        plt.subplot(335)
        val_means = [round(np.mean(scores).item()) for scores in scores["validation"]]
        plt.title(f"validation score: {val_means[-1]} out of {total_episode_rewards}")
        plt.plot(val_means)
        plt.xlabel("episode")

    if scores["test"]:
        plt.subplot(336)
        plt.title(
            f"test score: {np.mean(scores['test'])} out of {total_episode_rewards}"
        )
        plt.plot(round(np.mean(scores["test"]).item(), 2))
        plt.xlabel("episode")

    plt.subplot(331)
    plt.title("training td loss")
    plt.plot(training_loss)
    plt.xlabel("update counts")

    plt.subplot(332)
    plt.title("epsilons")
    plt.plot(epsilons)
    plt.xlabel("update counts")

    plt.subplot(337)
    plt.title("Q-values, train")
    for action_number in range(number_of_actions):
        plt.plot(
            [q_values_[action_number] for q_values_ in q_values["train"]],
            label=f"action {action_number}",
        )
    plt.legend(loc="upper left")
    plt.xlabel("number of actions")

    plt.subplot(338)
    plt.title("Q-values, val")
    for action_number in range(number_of_actions):
        plt.plot(
            [q_values_[action_number] for q_values_ in q_values["val"]],
            label=f"action {action_number}",
        )
    plt.legend(loc="upper left")
    plt.xlabel("number of actions")

    plt.subplot(339)
    plt.title("Q-values, test")
    for action_number in range(number_of_actions):
        plt.plot(
            [q_values_[action_number] for q_values_ in q_values["test"]],
            label=f"action {action_number}",
        )
    plt.legend(loc="upper left")
    plt.xlabel("number of actions")

    plt.subplots_adjust(hspace=0.5)
    plt.savefig(f"{default_root_dir}/plot.pdf")

    if is_notebook:
        plt.show()
    else:
        console_dqn(**all_params)


def console_dqn(
    scores: dict,
    training_loss: list,
    epsilons: list,
    q_values: dict,
    iteration_idx: int,
    number_of_actions: int,
    num_iterations: int,
    total_episode_rewards: int,
    num_validation: int,
    num_samples_for_results: int,
    default_root_dir: str,
) -> None:
    """Print the dqn training to the console."""
    if scores["train"]:
        tqdm.write(
            f"iteration {iteration_idx} out of {num_iterations}.\n"
            f"episode {num_validation} training score: "
            f"{scores['train'][-1]} out of {total_episode_rewards}"
        )

    if scores["validation"]:
        val_means = [round(np.mean(scores).item()) for scores in scores["validation"]]
        tqdm.write(
            f"episode {num_validation} validation score: {val_means[-1]} "
            "out of {total_episode_rewards}"
        )

    if scores["test"]:
        tqdm.write(
            f"test score: {np.mean(scores['test'])} out of {total_episode_rewards}"
        )

    # tqdm.write(
    #     f"training loss: {training_loss[-1]}\nepsilons: "
    #     f"{epsilons[-1]}\ntraining q-values: {q_values['train']}"
    # )

    tqdm.write(f"training loss: {training_loss[-1]}")


def save_dqn_results(
    scores: dict,
    training_loss: list,
    default_root_dir: str,
    q_values: dict,
    last_memory_state: dict,
) -> None:
    """Save dqn train / val / test results."""
    results = {
        "train_score": scores["train"],
        "validation_score": [
            {
                "mean": round(np.mean(scores).item(), 2),
                "std": round(np.std(scores).item(), 2),
            }
            for scores in scores["validation"]
        ],
        "test_score": {
            "mean": round(np.mean(scores["test"]).item(), 2),
            "std": round(np.std(scores["test"]).item(), 2),
        },
        "training_loss": training_loss,
    }
    write_yaml(results, os.path.join(default_root_dir, "results.yaml"))
    write_yaml(
        last_memory_state,
        os.path.join(default_root_dir, "last_memory_state_test.yaml"),
    )
    write_yaml(q_values, os.path.join(default_root_dir, "q_values.yaml"))


class ReplayBuffer:
    """A simple numpy replay buffer.

    numpy replay buffer is faster than deque or list.
    copied from https://github.com/Curt-Park/rainbow-is-all-you-need

    """

    def __init__(
        self,
        observation_type: str,
        size: int,
        obs_dim: tuple = None,
        batch_size: int = 32,
    ):
        """Initialize replay buffer.

        Args:
            observation_type: "dict" or "tensor"
            size: size of the buffer
            batch_size: batch size to sample

        """
        if batch_size > size:
            raise ValueError("batch_size must be smaller than size")
        if observation_type == "dict":
            self.obs_buf = np.array([{}] * size)
            self.next_obs_buf = np.array([{}] * size)
        else:
            raise ValueError("At the moment, observation_type must be 'dict'")
            # self.obs_buf = np.zeros([size, *obs_dim], dtype=np.float32)
            # self.next_obs_buf = np.zeros([size, *obs_dim], dtype=np.float32)

        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        (
            self.ptr,
            self.size,
        ) = (
            0,
            0,
        )

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size


class ReplayBufferNStep:
    """A simple numpy N step replay buffer.
    copied from https://github.com/Curt-Park/rainbow-is-all-you-need

    """

    def __init__(
        self,
        observation_type: str,
        size: int,
        obs_dim: tuple = None,
        batch_size: int = 32,
        n_step: int = 1,
        gamma: float = 0.99,
    ):
        """Initialize replay buffer.

        Args:
            observation_type: "dict" or "tensor"
            size: size of the buffer
            batch_size: batch size to sample

        """
        if observation_type == "dict":
            self.obs_buf = np.array([{}] * size)
            self.next_obs_buf = np.array([{}] * size)
        else:
            self.obs_buf = np.zeros([size, *obs_dim], dtype=np.float32)
            self.next_obs_buf = np.zeros([size, *obs_dim], dtype=np.float32)

        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        (
            self.ptr,
            self.size,
        ) = (
            0,
            0,
        )

        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()

        # make a n-step transition
        rew, next_obs, done = self._get_n_step_info(self.n_step_buffer, self.gamma)
        obs, act = self.n_step_buffer[0][:2]

        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        return self.n_step_buffer[0]

    def sample_batch(self) -> dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)

        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
            # for N-step Learning
            indices=idxs,
        )

    def sample_batch_from_idxs(self, idxs: np.ndarray) -> dict[str, np.ndarray]:
        # for N-step Learning
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

    def _get_n_step_info(
        self, n_step_buffer: Deque, gamma: float
    ) -> tuple[np.int64, np.ndarray, bool]:
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        rew, next_obs, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBufferNStep):
    """Prioritized Replay buffer.

    copied from https://github.com/Curt-Park/rainbow-is-all-you-need


    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight

    """

    def __init__(
        self,
        observation_type: str,
        size: int,
        obs_dim: tuple = None,
        batch_size: int = 32,
        alpha: float = 0.6,
        n_step: int = 1,
        gamma: float = 0.99,
    ):
        """Initialization."""
        assert alpha >= 0

        super(PrioritizedReplayBuffer, self).__init__(
            observation_type, size, obs_dim, batch_size, n_step, gamma
        )
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(
        self,
        obs: np.ndarray,
        act: int,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        """Store experience and priority."""
        transition = super().store(obs, act, rew, next_obs, done)

        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority**self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority**self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size

        return transition

    def sample_batch(self, beta: float = 0.4) -> dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()

        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,
        )

    def update_priorities(self, indices: list[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority**self.alpha
            self.min_tree[idx] = priority**self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self) -> list[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight


def compute_dqn_loss(
    samples: dict[str, np.ndarray],
    device: str,
    dqn: torch.nn.Module,
    dqn_target: torch.nn.Module,
    ddqn: str,
    gamma: float,
) -> torch.Tensor:
    """Return dqn loss.

    Args:
        samples: A dictionary of samples from the replay buffer.
            obs: np.ndarray,
            act: np.ndarray,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
        device:
        dqn:
        dqn_target:
        ddqn:
        gamma:

    Returns:
        loss: torch.Tensor

    """
    state = samples["obs"]
    next_state = samples["next_obs"]
    action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
    reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
    done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

    # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
    #       = r                       otherwise
    curr_q_value = dqn(state).gather(1, action)
    if ddqn:
        next_q_value = (
            dqn_target(next_state)
            .gather(1, dqn(next_state).argmax(dim=1, keepdim=True))
            .detach()
        )
    else:
        next_q_value = dqn_target(next_state).max(dim=1, keepdim=True)[0].detach()
    mask = 1 - done
    target = (reward + gamma * next_q_value * mask).to(device)

    # calculate dqn loss
    loss = F.smooth_l1_loss(curr_q_value, target)

    return loss


def select_dqn_action(
    state: dict,
    greedy: bool,
    dqn: torch.nn.Module,
    train_val_test: str,
    q_values: dict,
    epsilon: float,
    action_space: gym.spaces.Discrete,
    save_q_value: bool = False,
) -> int:
    """Select an action from the input state.

    Args:
        state: The current state of the memory systems. This is NOT what the gym env
        gives you. This is made by the agent.
        greedy: always pick greedy action if True
        save_q_value: whether to save the q values or not.

    """
    # epsilon greedy policy
    q_values_ = dqn(np.array([state])).detach().cpu().numpy().tolist()[0]

    if save_q_value:
        if train_val_test == "train":
            q_values["train"].append(q_values_)
        elif train_val_test == "val":
            q_values["val"].append(q_values_)
        elif train_val_test == "test":
            q_values["test"].append(q_values_)

    if epsilon < np.random.random() or greedy:
        selected_action = argmax(q_values_)
    else:
        selected_action = action_space.sample()

    return selected_action


def update_dqn_model(
    replay_buffer: ReplayBuffer,
    optimizer: torch.optim.Adam,
    device: str,
    dqn: torch.nn.Module,
    dqn_target: torch.nn.Module,
    ddqn: str,
    gamma: float,
) -> torch.Tensor:
    """Update the model by gradient descent."""
    samples = replay_buffer.sample_batch()

    loss = compute_dqn_loss(samples, device, dqn, dqn_target, ddqn, gamma)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def save_dqn_validation(
    scores_temp: list,
    scores: dict,
    default_root_dir: str,
    num_validation: int,
    val_filenames: list,
    dqn: torch.nn.Module,
) -> None:
    """Keep the best validation model.

    Args:
        scores_temp: a list of validation scores for the current validation episode.
        scores: a dictionary of scores for train, validation, and test.
        default_root_dir: the root directory where the results are saved.
        num_validation: the current validation episode.
        val_filenames: a list of filenames for the validation models.
        dqn: the dqn model.

    """
    mean_score = round(np.mean(scores_temp).item())
    filename = (
        f"{default_root_dir}/" f"episode={num_validation}_val-score={mean_score}.pt"
    )
    val_filenames.append(filename)
    torch.save(dqn.state_dict(), filename)
    scores["validation"].append(scores_temp)

    scores_to_compare = []
    for filename in val_filenames:
        scores_to_compare.append(int(filename.split("val-score=")[-1].split(".pt")[0]))

    file_to_keep = val_filenames[scores_to_compare.index(max(scores_to_compare))]

    for filename in deepcopy(val_filenames):
        if filename != file_to_keep:
            os.remove(filename)
            val_filenames.remove(filename)


def dqn_target_hard_update(dqn: torch.nn.Module, dqn_target: torch.nn.Module) -> None:
    """Hard update: target <- local."""
    dqn_target.load_state_dict(dqn.state_dict())
