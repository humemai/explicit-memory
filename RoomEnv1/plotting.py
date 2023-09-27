"""Matplotlib functions."""
import logging

logger = logging.getLogger()
logger.disabled = True

import os
from copy import deepcopy
from glob import glob
from typing import List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import room_env
import torch
from room_env.utils import get_handcrafted
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from train import DQNLightning, RLAgent


def load_training_val_test_results(
    data_dir: str = "./data/v1-question_prob=1.0/",
    kind: str = "train_total_reward",
    capacity: int = 32,
    pretrain: bool = False,
    des_size: str = "l",
) -> np.ndarray:
    assert kind in [
        "train_loss",
        "train_total_reward",
        "val_total_reward_mean",
        "test_total_reward_mean",
    ]
    paths = glob(
        os.path.join(
            data_dir,
            kind,
            "*.csv",
        )
    )
    paths = [
        path
        for path in paths
        if f"capacity={capacity}" in path
        and (
            (f"pretrain_semantic={pretrain}" in path)
            or (f"pretrain={pretrain}" in path)
        )
        and f"des_size={des_size}" in path
    ]
    assert len(paths) == 5
    runs = []
    for path in paths:
        df = pd.read_csv(path)
        values = df.Value.to_numpy()
        if "train" in kind or "val" in kind:
            if "reward" in kind:
                assert len(values) == 16
            else:
                assert len(values) == 1000
        elif "test" in kind:
            assert len(values) == 1
        else:
            raise ValueError
        runs.append(values)

    runs = np.array(runs)
    means = np.mean(runs, axis=0)
    stds = np.std(runs, axis=0)
    steps = df.Step.to_numpy()

    assert len(means) == len(stds) == len(steps)

    if len(means) == 1:
        return {"mean": means.item(), "std": stds.item(), "step": steps.item()}
    else:
        return {"means": means, "stds": stds, "steps": steps}


def load_episodic_semantic_random_scratch_pretrained(
    data_dir: str = "./data/v1-question_prob=1.0/",
    kind: str = "test_total_reward_mean",
    capacity: int = 32,
    des_size: str = "l",
    question_prob: float = 1.0,
    allow_random_human: str = False,
    allow_random_question: str = False,
) -> dict:
    results = get_handcrafted(
        env="RoomEnv-v1",
        des_size=des_size,
        seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        question_prob=question_prob,
        policies={
            "memory_management": "rl",
            "answer_question": "episodic_semantic",
            "encoding": "argmax",
        },
        capacities=[capacity],
        allow_random_human=allow_random_human,
        allow_random_question=allow_random_question,
        varying_rewards=False,
        check_resources=True,
    )
    results = results[capacity]

    del results["pre_sem"]

    results["scratch"] = load_training_val_test_results(
        data_dir=data_dir,
        kind=kind,
        capacity=capacity,
        pretrain=False,
        des_size=des_size,
    )
    results["pretrained"] = load_training_val_test_results(
        data_dir=data_dir,
        kind=kind,
        capacity=capacity,
        pretrain=True,
        des_size=des_size,
    )

    return results


def plot_training_validation_results(
    data_dir: str = "./data/v1-question_prob=1.0/",
    kind: str = "train_total_reward",
    capacity: int = 32,
    save_dir: str = "./figures/",
    ymin: int = 64,
    ymax: int = 128,
    ylog: bool = False,
    xlabel: str = "Step",
    des_size: str = "l",
    figsize: Tuple = (10, 10),
    legend_loc: str = "upper left",
) -> None:
    assert kind in [
        "train_loss",
        "train_total_reward",
        "val_total_reward_mean",
    ]

    if kind == "train_loss":
        title = "Avg. loss, training."
        ylabel = "Avg. loss"

    elif kind == "train_total_reward":
        title = "Avg. total rewards, training."
        ylabel = "Avg. total rewards"

    elif kind == "val_total_reward_mean":
        title = "Avg. total rewards, validation."
        ylabel = "Avg. total rewards"

    else:
        raise ValueError

    fig, ax = plt.subplots(figsize=figsize)

    training_val_test_results = load_training_val_test_results(
        data_dir=data_dir,
        kind=kind,
        capacity=capacity,
        pretrain=False,
        des_size=des_size,
    )
    means, stds, steps = (
        training_val_test_results["means"],
        training_val_test_results["stds"],
        training_val_test_results["steps"],
    )
    # print(kind, "pretrain=False", means, stds, steps)
    ax.plot(steps, means, color="pink")
    ax.fill_between(
        steps,
        means - stds,
        means + stds,
        alpha=0.2,
        edgecolor="pink",
        facecolor="pink",
        label="_nolegend_",
    )

    training_val_test_results = load_training_val_test_results(
        data_dir=data_dir,
        kind=kind,
        capacity=capacity,
        pretrain=True,
        des_size=des_size,
    )
    means, stds, steps = (
        training_val_test_results["means"],
        training_val_test_results["stds"],
        training_val_test_results["steps"],
    )
    # print(kind, "pretrain=False", f"means: {means}")
    ax.plot(steps, means, color="deeppink")
    ax.fill_between(
        steps,
        means - stds,
        means + stds,
        alpha=0.2,
        edgecolor="deeppink",
        facecolor="deeppink",
        label="_nolegend_",
    )
    if ymin is None and ymax is None:
        ymin, ymax = ax.get_ylim()

    ax.set_ylim(ymin, ymax)
    # ax.yaxis.set_ticks(np.arange(ymin, ymax, 7))

    if ylog:
        plt.yscale("log")
        ylabel += " (log scale)"
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=30)
    ax.set_xlabel(xlabel, fontsize=35)
    ax.set_ylabel(ylabel, fontsize=35)
    ax.legend(
        ["Semantic memory from scratch", "Semantic memory prefilled"],
        fontsize=25,
        loc=legend_loc,
    )

    plt.title(title, fontsize=40)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if "v1" in data_dir:
        filename = f"des_size={des_size}-capacity={capacity}-{kind}-v1.pdf"
    else:
        filename = f"des_size={des_size}-capacity={capacity}-{kind}.pdf"

    plt.savefig(os.path.join(save_dir, filename))


def plot_test_results(
    data_dir: str = "./data/v1-question_prob=1.0/",
    capacity: int = 32,
    save_dir: str = "./figures/",
    ymin: int = 64,
    ymax: int = 128,
    des_size: str = "l",
    figsize: Tuple = (10, 10),
    legend_loc: str = "upper left",
    question_prob: float = 1.0,
    allow_random_human: str = False,
    allow_random_question: str = False,
) -> None:
    results = load_episodic_semantic_random_scratch_pretrained(
        data_dir=data_dir,
        kind="test_total_reward_mean",
        capacity=capacity,
        des_size=des_size,
        question_prob=question_prob,
        allow_random_human=allow_random_human,
        allow_random_question=allow_random_question,
    )

    fig, ax = plt.subplots(figsize=figsize)
    title = "Avg. total rewards, test."
    ylabel = "Avg. total rewards"

    legend_order = [
        "Episodic only, handcrafted",
        "Semantic only, handcrafted",
        "Both, random",
        "Both, RL, semantic from scratch",
        "Both, RL, semantic pretrained",
    ]
    color_order = ["orange", "dodgerblue", "gray", "pink", "deeppink"]
    agent_order = ["episodic", "semantic", "random", "scratch", "pretrained"]

    for legend, color, agent in zip(legend_order, color_order, agent_order):
        height = results[agent]["mean"]
        yerr = results[agent]["std"]
        ax.bar(
            x=legend,
            height=height,
            color=color,
            width=0.9,
            yerr=yerr,
            capsize=4,
        )
        # print(f"heights: {height}, stds: {yerr}")
    plt.xticks([])
    ax.set_ylim([ymin, ymax])
    plt.yticks(fontsize=30)
    # ax.yaxis.set_ticks(np.arange(ymin, ymax, 7))

    ax.legend(legend_order, fontsize=20, loc=legend_loc)
    ax.set_xlabel("Agent type", fontsize=35)
    ax.set_ylabel(ylabel, fontsize=35)
    plt.title(title, fontsize=40)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if "v1" in data_dir:
        filename = (
            f"des_size={des_size}-capacity={capacity}-test_total_reward_mean-v1.pdf"
        )
    else:
        raise ValueError

    plt.savefig(os.path.join(save_dir, filename))


def plot_test_results_all_capacities(
    data_dir: str = "./data/v1-question_prob=1.0/",
    save_dir: str = "./figures/",
    ymin: int = 0,
    ymax: int = 128,
    des_size: str = "l",
    figsize: Tuple = (25, 8),
    legend_loc: str = "upper left",
    question_prob: float = 1.0,
    allow_random_human: str = False,
    allow_random_question: str = False,
) -> None:
    capacities = [2, 4, 8, 16, 32, 64]
    width = 0.17

    results = {}
    for capacity in capacities:
        results_ = load_episodic_semantic_random_scratch_pretrained(
            data_dir=data_dir,
            kind="test_total_reward_mean",
            capacity=capacity,
            des_size=des_size,
            question_prob=question_prob,
            allow_random_human=allow_random_human,
            allow_random_question=allow_random_question,
        )
        results[capacity] = results_

    idx = np.asanyarray([i for i in range(len(results))])

    fig, ax = plt.subplots(figsize=figsize)
    title = "Avg. total rewards, varying capacities, test."
    ylabel = "Avg. total rewards"

    legend_order = [
        "Episodic only, handcrafted",
        "Semantic only, handcrafted",
        "Both, random",
        "Both, RL, semantic from scratch",
        "Both, RL, semantic pretrained",
    ]

    color_order = ["orange", "dodgerblue", "gray", "pink", "deeppink"]
    agent_order = ["episodic", "semantic", "random", "scratch", "pretrained"]

    for w, color, agent in zip([-2, -1, 0, 1, 2], color_order, agent_order):
        height = [results_[agent]["mean"] for _, results_ in results.items()]
        yerr = [results_[agent]["std"] for _, results_ in results.items()]
        ax.bar(
            x=idx + w * width,
            height=height,
            yerr=yerr,
            width=width,
            color=color,
            capsize=4,
        )
        # print(f"agent: {agent}, height: {height}, std: {yerr}")
    ax.set_xticks(idx)
    ax.set_xticklabels(list(results.keys()))
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=30)
    ax.set_ylim([ymin, ymax])
    ax.legend(legend_order, fontsize=30, loc=legend_loc)
    ax.set_xlabel("Memory capacity", fontsize=40)
    ax.set_ylabel(ylabel, fontsize=40)
    plt.title(title, fontsize=50)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if "v1" in data_dir:
        filename = f"des_size={des_size}-capacity=all-test_total_reward_mean-v1.pdf"
    else:
        raise ValueError

    plt.savefig(os.path.join(save_dir, filename))

    plt.savefig(os.path.join(save_dir, filename))


class UnderstandModel:
    def __init__(
        self,
        model_scratch_path: str = "./models/v1-question_prob=1.0/allow_random_human=False_allow_random_question=False_pretrain_semantic=False_varying_rewards=False_des_size=l_capacity=32_question_prob=1.0_seed=2/checkpoints/epoch=13-val_total_reward_mean=44.00-val_total_reward_std=7.87.ckpt",
        model_pretrained_path: str = "./models/v1-question_prob=1.0/allow_random_human=False_allow_random_question=False_pretrain_semantic=True_varying_rewards=False_des_size=l_capacity=32_question_prob=1.0_seed=3/checkpoints/epoch=05-val_total_reward_mean=51.60-val_total_reward_std=7.23.ckpt",
    ) -> None:
        """Call the model loading."""
        self.model_scratch_path = model_scratch_path
        self.model_pretrained_path = model_pretrained_path

        self._load_models()

        self.indices = {
            interest: {
                i: key
                for i, (key, val) in enumerate(
                    self.embeddings["scratch"][interest].items()
                )
            }
            for interest in ["humans", "objects", "object_locations"]
        }

    def _load_models(self) -> None:
        """Load the models and its embeddings.

        I don't know why but this takes some time.
        """
        self.models = {}
        self.embeddings = {}
        for semantic in ["scratch", "pretrained"]:
            if semantic == "scratch":
                model_path = self.model_scratch_path
            else:
                model_path = self.model_pretrained_path

            self.models[semantic] = DQNLightning.load_from_checkpoint(model_path)
            self.models[semantic].eval()

            self.word2idx = self.models[semantic].net.word2idx

            self.embeddings[semantic] = {}

            self.embeddings[semantic]["humans"] = {
                human: self.models[semantic]
                .net.embeddings(torch.tensor(self.word2idx[human]))
                .detach()
                .cpu()
                .numpy()
                for human in self.models[semantic].env.des.humans
            }

            self.embeddings[semantic]["objects"] = {
                obj: self.models[semantic]
                .net.embeddings(torch.tensor(self.word2idx[obj]))
                .detach()
                .cpu()
                .numpy()
                for obj in self.models[semantic].env.des.objects
            }

            self.embeddings[semantic]["object_locations"] = {
                obj_loc: self.models[semantic]
                .net.embeddings(torch.tensor(self.word2idx[obj_loc]))
                .detach()
                .cpu()
                .numpy()
                for obj_loc in self.models[semantic].env.des.object_locations
            }

    def compute_reduction(
        self,
        tsne_params: dict = None,
        pca_params: dict = None,
    ) -> None:
        """Compute dimension reduction on the embeddings.

        Args
        ----
        tsne_params: t-SNE parameters, e.g., {"n_components": 2, "perplexity": 3}
            This is a non-linear reduction method. See the paper.
        pca_params: PCA parameters, e.g., {"n_components": 2}
            This is a linear reduction method. See wikipedia.

        """
        if tsne_params is None:
            assert pca_params is not None
            self.reduction_method = "PCA"
        if pca_params is None:
            assert tsne_params is not None
            self.reduction_method = "t-SNE"

        self.X = {}
        for semantic in ["scratch", "pretrained"]:
            self.X[semantic] = {}
            for interest in ["humans", "objects", "object_locations"]:
                X = np.array(list(self.embeddings[semantic][interest].values()))

                if self.reduction_method == "t-SNE":
                    self.X[semantic][interest] = TSNE(**tsne_params).fit_transform(X)
                else:
                    pca = PCA(**pca_params)
                    pca.fit(X)
                    self.X[semantic][interest] = pca.transform(X)

    def plot_embeddings(
        self,
        semantic: str = "scratch",
        interest: str = "humans",
        figsize: Tuple = (10, 10),
        save_dir: str = "./figures/",
    ) -> None:
        self.semantic = semantic
        self.interest = interest
        if semantic == "scratch":
            color = "pink"
        else:
            color = "deeppink"
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(
            self.X[self.semantic][self.interest][:, 0],
            self.X[self.semantic][self.interest][:, 1],
            color=color,
            edgecolors="black",
            s=250,
        )
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)

        ax.set_xlabel(f"{self.reduction_method} dimension0", fontsize=35)
        ax.set_ylabel(f"{self.reduction_method} dimension1", fontsize=35)

        plt.title(
            f"{self.interest} embeddings\n" f"semantic-{semantic}",
            fontsize=40,
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(
            os.path.join(
                save_dir, f"{self.reduction_method}-{self.semantic}-{self.interest}.pdf"
            )
        )

    def get_similar_embeddings(
        self,
        xmin: int,
        xmax: int,
        ymin: int,
        ymax: int,
    ) -> List:
        condition_xmin = xmin < self.X[self.semantic][self.interest][:, 0]
        condition_xmax = xmax > self.X[self.semantic][self.interest][:, 0]
        condition_ymin = ymin < self.X[self.semantic][self.interest][:, 1]
        condition_ymax = ymax > self.X[self.semantic][self.interest][:, 1]

        condition = condition_xmin * condition_xmax * condition_ymin * condition_ymax
        idx_embs = np.where(condition)[0]

        entities = [self.indices[self.interest][idx] for idx in idx_embs]

        return entities
