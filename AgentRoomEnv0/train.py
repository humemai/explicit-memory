import logging

logger = logging.getLogger()
logger.disabled = True

import argparse
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import yaml
from agent import HandcraftedAgent
from tqdm import tqdm

from explicit_memory.utils import is_running_notebook

is_notebook = is_running_notebook()

parser = argparse.ArgumentParser(description="train")
parser.add_argument(
    "-c",
    "--config",
    default="./train.yaml",
    type=str,
    help="config file path (default: ./train.yaml)",
)
args = parser.parse_args()
with open(args.config, "r") as stream:
    env_params = yaml.safe_load(stream)
print("Arguments:")
for k, v in env_params.items():
    print(f"  {k:>21} : {v}")


with open("./train.yaml", "r") as stream:
    env_params = yaml.safe_load(stream)

results_all = {}
seeds = [0, 1, 2, 3, 4]
caps = [4, 8, 16, 32]
num_agentss = [1]
agent_types = [
    "episodic",
    "semantic",
    "episodic_semantic",
    "episodic_semantic_pretrain",
]

for agent_type in tqdm(agent_types):
    results_all[agent_type] = {}
    strategy_types = []
    if agent_type == "episodic":
        for forget_policy in ["oldest", "random"]:
            for answer_policy in ["latest", "random"]:
                strategy_types.append(f"{forget_policy}_{answer_policy}")

    elif agent_type == "semantic":
        for forget_policy in ["weakest", "random"]:
            for answer_policy in ["strongest", "random"]:
                strategy_types.append(f"{forget_policy}_{answer_policy}")
    elif agent_type == "episodic_semantic":
        for forget_policy in ["generalize", "random"]:
            for answer_policy in ["episem", "random"]:
                strategy_types.append(f"{forget_policy}_{answer_policy}")
    else:
        for forget_policy in ["oldest", "random"]:
            for answer_policy in ["episem", "random"]:
                strategy_types.append(f"{forget_policy}_{answer_policy}")

    for capacity in caps:
        results_all[agent_type][capacity] = {}
        for strategy_type in strategy_types:
            results_all[agent_type][capacity][strategy_type] = {}
            for num_agents in num_agentss:
                stats = []
                for seed in seeds:
                    forget_policy, answer_policy = strategy_type.split("_")
                    if agent_type == "episodic":
                        episodic_capacity = capacity
                        semantic_capacity = 0
                    elif agent_type == "semantic":
                        episodic_capacity = 0
                        semantic_capacity = capacity
                    else:
                        episodic_capacity = capacity // 2
                        semantic_capacity = capacity // 2

                    env_params["num_agents"] = num_agents
                    env = gym.make("room_env:RoomEnv-v0", **env_params)
                    agent = HandcraftedAgent(
                        seed=seed,
                        agent_type=agent_type,
                        forget_policy=forget_policy,
                        answer_policy=answer_policy,
                        episodic_capacity=episodic_capacity,
                        semantic_capacity=semantic_capacity,
                        env=env,
                    )
                    agent.run()
                    stats.append(agent.rewards)
                results_all[agent_type][capacity][strategy_type][num_agents] = {
                    "mean": round(np.mean(stats).item(), 4),
                    "std": round(np.std(stats).item(), 4),
                }

for num_agents in num_agentss:
    fig, ax = plt.subplots(figsize=(10, 5))
    idx = np.asanyarray([i for i in range(len(caps))])
    width = 0.2

    legend_order = [
        "Hand-crafted 1: Only episodic",
        "Hand-crafted 2: Only semantic",
        "Hand-crafted 3: Both epi and sem",
        "Hand-crafted 4: Both epi and pre-sem",
    ]

    color_order = ["orange", "dodgerblue", "yellowgreen", "deeppink"]

    for agent_type, w, color in zip(agent_types, [-1.5, -0.5, 0.5, 1.5], color_order):
        if agent_type == "episodic":
            strategy_type = "oldest_latest"
        elif agent_type == "semantic":
            strategy_type = "weakest_strongest"
        elif agent_type == "episodic_semantic":
            strategy_type = "generalize_episem"
        else:
            strategy_type = "oldest_episem"

        means = [
            results_all[agent_type][capacity][strategy_type][num_agents]["mean"]
            for capacity in caps
        ]
        stds = [
            results_all[agent_type][capacity][strategy_type][num_agents]["std"]
            for capacity in caps
        ]

        print(agent_type, means)
        ax.bar(
            x=idx + w * width,
            height=means,
            yerr=stds,
            width=width,
            color=color,
            capsize=2,
        )
        ax.set_xticks(idx)
        ax.set_xticklabels(caps)
        ax.legend(legend_order, fontsize=10, loc="upper left")
        ax.set_xlabel("memory capacity", fontsize=15)
        ax.set_ylabel(f"total rewards ({len(seeds)} random seeds)", fontsize=15)

        ax.set_ylim([0, env_params["max_step"]])

        fig.tight_layout()
        plt.title(
            f"Best performance by memory type, number of agents={num_agents}",
            fontsize=15,
        )
        os.makedirs("figures", exist_ok=True)
        for fmt in ["png", "pdf"]:
            plt.savefig(
                f"./figures/best-strategies-{num_agents}-v0.{fmt}", bbox_inches="tight"
            )
        if is_notebook:
            plt.show()

idx = np.asanyarray([i for i in range(len(caps))])
width = 0.2

for num_agents in num_agentss:
    for agent_type in agent_types:
        fig, ax = plt.subplots(figsize=(10, 5))

        if agent_type == "episodic":
            strategy_order = [
                "oldest_latest",
                "random_latest",
                "oldest_random",
                "random_random",
            ]

            legend_order = [
                "forget oldest, answer latest",
                "forget random, answer latest",
                "forget oldest, answer random",
                "forget random, answer random",
            ]

            color_order = ["orange", "navajowhite", "blanchedalmond", "oldlace"]

        elif agent_type == "semantic":
            strategy_order = [
                "weakest_strongest",
                "random_strongest",
                "weakest_random",
                "random_random",
            ]

            legend_order = [
                "forget weakest, answer strongest",
                "forget random, answer strongest",
                "forget weakest, answer random",
                "forget random, answer random",
            ]

            color_order = ["dodgerblue", "lightskyblue", "powderblue", "aliceblue"]

        elif agent_type == "episodic_semantic":
            strategy_order = [
                "generalize_episem",
                "random_episem",
                "generalize_random",
                "random_random",
            ]

            legend_order = [
                "forget by generalization, answer epi and then sem",
                "forget random, answer epi and then sem",
                "forget by generalization, answer random",
                "forget random, answer random",
            ]

            color_order = ["yellowgreen", "lightgreen", "palegreen", "honeydew"]

        elif agent_type == "episodic_semantic_pretrain":
            strategy_order = [
                "oldest_episem",
                "random_episem",
                "oldest_random",
                "random_random",
            ]

            legend_order = [
                "forget oldest, answer epi and then sem",
                "forget random, answer epi and then sem",
                "forget oldest, answer random",
                "forget random, answer random",
            ]

            color_order = ["deeppink", "lightpink", "pink", "lavenderblush"]

        else:
            raise ValueError

        for strategy_type, w, color in zip(
            strategy_order, [-1.5, -0.5, 0.5, 1.5], color_order
        ):
            means = [
                results_all[agent_type][capacity][strategy_type][num_agents]["mean"]
                for capacity in caps
            ]
            stds = [
                results_all[agent_type][capacity][strategy_type][num_agents]["std"]
                for capacity in caps
            ]

            print(strategy_type, means)
            ax.bar(
                x=idx + w * width,
                height=means,
                yerr=stds,
                width=width,
                color=color,
                capsize=2,
            )
            ax.set_xticks(idx)
            ax.set_xticklabels(caps)
            ax.legend(legend_order, fontsize=10, loc="upper left")
            ax.set_xlabel("memory capacity", fontsize=15)
            ax.set_ylabel(f"total rewards ({len(seeds)} random seeds)", fontsize=15)

            ax.set_ylim([0, env_params["max_step"]])

            fig.tight_layout()
            if agent_type == "episodic":
                title = f"Only episodic, performance by strategy, number of agents={num_agents}"
            elif agent_type == "semantic":
                title = f"Only semantic, performance by strategy, number of agents={num_agents}"
            elif agent_type == "episodic_semantic":
                title = f"Both episodic and semantic, performance by strategy, number of agents={num_agents}"
            elif agent_type == "episodic_semantic_pretrain":
                title = f"Both episodic and pretrained semantic, performance by strategy, number of agents={num_agents}"
            else:
                raise ValueError

            plt.title(title, fontsize=15)
            for fmt in ["png", "pdf"]:
                plt.savefig(
                    f"./figures/{agent_type}-{num_agents}-v0.{fmt}", bbox_inches="tight"
                )

results_all = {}
seeds = [0, 1, 2, 3, 4]
caps = [16, 32]
num_agentss = [1, 2]
agent_types = [
    "episodic",
    "semantic",
    "episodic_semantic",
    "episodic_semantic_pretrain",
]

for agent_type in tqdm(agent_types):
    results_all[agent_type] = {}
    strategy_types = []
    if agent_type == "episodic":
        for forget_policy in ["oldest", "random"]:
            for answer_policy in ["latest", "random"]:
                strategy_types.append(f"{forget_policy}_{answer_policy}")

    elif agent_type == "semantic":
        for forget_policy in ["weakest", "random"]:
            for answer_policy in ["strongest", "random"]:
                strategy_types.append(f"{forget_policy}_{answer_policy}")
    elif agent_type == "episodic_semantic":
        for forget_policy in ["generalize", "random"]:
            for answer_policy in ["episem", "random"]:
                strategy_types.append(f"{forget_policy}_{answer_policy}")
    else:
        for forget_policy in ["oldest", "random"]:
            for answer_policy in ["episem", "random"]:
                strategy_types.append(f"{forget_policy}_{answer_policy}")

    for capacity in caps:
        results_all[agent_type][capacity] = {}
        for strategy_type in strategy_types:
            results_all[agent_type][capacity][strategy_type] = {}
            for num_agents in num_agentss:
                stats = []
                for seed in seeds:
                    forget_policy, answer_policy = strategy_type.split("_")
                    if agent_type == "episodic":
                        episodic_capacity = capacity
                        semantic_capacity = 0
                    elif agent_type == "semantic":
                        episodic_capacity = 0
                        semantic_capacity = capacity
                    else:
                        episodic_capacity = capacity // 2
                        semantic_capacity = capacity // 2

                    env_params["num_agents"] = num_agents
                    env = gym.make("room_env:RoomEnv-v0", **env_params)
                    agent = HandcraftedAgent(
                        seed=seed,
                        agent_type=agent_type,
                        forget_policy=forget_policy,
                        answer_policy=answer_policy,
                        episodic_capacity=episodic_capacity,
                        semantic_capacity=semantic_capacity,
                        env=env,
                    )
                    agent.run()
                    stats.append(agent.rewards)
                results_all[agent_type][capacity][strategy_type][num_agents] = {
                    "mean": round(np.mean(stats).item(), 4),
                    "std": round(np.std(stats).item(), 4),
                }


caps = {1: [32], 2: [16]}

fig, ax = plt.subplots(figsize=(10, 5))
color_order = {
    1: ["navajowhite", "lightskyblue", "lightgreen", "lightpink"],
    2: ["orange", "dodgerblue", "yellowgreen", "deeppink"],
}
width = {1: 0.1, 2: 0.15}

legend_order = [
    "Hand-crafted 1: Only episodic",
    "Hand-crafted 2: Only semantic",
    "Hand-crafted 3: Both epi and sem",
    "Hand-crafted 4: Both epi and pre-sem",
]
for num_agents in num_agentss[::-1]:
    idx = np.asanyarray([i for i in range(len(caps[num_agents]))])

    for agent_type, w, color in zip(
        agent_types, [-1.5, -0.5, 0.5, 1.5], color_order[num_agents]
    ):
        if agent_type == "episodic":
            strategy_type = "oldest_latest"
        elif agent_type == "semantic":
            strategy_type = "weakest_strongest"
        elif agent_type == "episodic_semantic":
            strategy_type = "generalize_episem"
        else:
            strategy_type = "oldest_episem"

        means = [
            results_all[agent_type][capacity][strategy_type][num_agents]["mean"]
            for capacity in caps[num_agents]
        ]
        stds = [
            results_all[agent_type][capacity][strategy_type][num_agents]["std"]
            for capacity in caps[num_agents]
        ]

        print(agent_type, means)
        ax.bar(
            x=idx + w * 0.2,
            height=means,
            yerr=stds,
            width=width[num_agents],
            color=color,
            capsize=2,
        )
        ax.set_xticks([])
        # ax.set_xticklabels(caps[num_agents])
        ax.legend(legend_order, fontsize=10, loc="upper left")
        ax.set_xlabel("Memory capacity is 32 for both cases.", fontsize=15)
        ax.set_ylabel(f"total rewards ({len(seeds)} random seeds)", fontsize=15)

        ax.set_ylim([0, env_params["max_step"]])

        fig.tight_layout()
        plt.title(
            f"Performance gap between a single agent and double agents",
            fontsize=15,
        )
        os.makedirs("figures", exist_ok=True)
        for fmt in ["png", "pdf"]:
            plt.savefig(
                f"./figures/single-and-double-agents-v0.{fmt}", bbox_inches="tight"
            )
        if is_notebook:
            plt.show()
